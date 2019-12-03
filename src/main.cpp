/**
 * main.cpp
 */

#include "main.h"
#include <cmath>

static char MODULE_INPUT[] = "input";
static char MODULE_GLOBAL[] = "global";
static char MODULE_MODEL[] = "model";

model_config_t config;
cli_arguments_t cli;
ImageSet trainImageSet;
LabelSet trainLabelSet;
ImageSet testImageSet;
LabelSet testLabelSet;
ModelFacade model;
LayerFacade layers[8];

void printPredictOutput(LayerFacade& layer, int depth = 0) {
    if (layer.schema->type == LAYER_TYPE_DENSE) {
        int d = layer.schema->outputDepth;
        d = d < 20 ? d : 20;
        printMatrixlf8(layer.predictOutput, d, 1);
    } else if (layer.schema->type == LAYER_TYPE_OUTPUT) {
        int outputSize = layer.schema->outputDepth * layer.schema->outputHeight * layer.schema->outputWidth;
        int batchSize = 100;
        double* y = layer.predictTemp + batchSize * 1 + batchSize * 1 + batchSize * outputSize;
        printMatrixlf8(y, outputSize, 1);
    } else if (layer.schema->type == LAYER_TYPE_CONVOLUTION) {
        int size = layer.schema->outputWidth * layer.schema->outputHeight;
        printMatrixlf8(layer.predictOutput + depth * size, layer.schema->outputWidth, layer.schema->outputHeight);
    }
}

void printPredictInput(LayerFacade& layer, int depth = 0) {
    int size = layer.schema->inputWidth * layer.schema->inputHeight;
    printMatrixlf8(layer.predictInput + depth * size, layer.schema->inputWidth, layer.schema->inputHeight);
}

void printWeights(LayerFacade& layer, int depth = 0) {
    if (layer.schema->type == LAYER_TYPE_CONVOLUTION) {
        int size = layer.schema->inputDepth * layer.schema->operationWidth * layer.schema->operationHeight + 1;
        printf("b[%d] = %.8lf, w[%d] = \n", depth, layer.weights[depth * size], depth);
        printMatrixlf8(layer.weights + depth * size + 1, layer.schema->operationWidth, layer.schema->operationHeight);;
    }
}

void printDweights(LayerFacade& layer, int depth = 0) {
    if (layer.schema->type == LAYER_TYPE_CONVOLUTION) {
        int size = layer.schema->inputDepth * layer.schema->operationWidth * layer.schema->operationHeight + 1;
        printf("db[%d] = %.8lf, dw[%d] = \n", depth, layer.dweights[depth * size], depth);
        printMatrixlf8(layer.dweights + depth * size + 1, layer.schema->operationWidth, layer.schema->operationHeight);;
    }
}

void printConvolutionLayer(LayerFacade& layer) {
    printPredictInput(layer);
    printWeights(layer);
    printWeights(layer, 1);
    printPredictOutput(layer);
    printPredictOutput(layer, 1);
}

int trainListener(model_schema_t* mem, int batchIndex, int step) {
    // layers[1].read();
    if (step == 1) {
        // for (int i = 1; i <= 7; i++) {
        //     layers[i].read();
        //     printPredictOutput(layers[i]);
        // }

        layers[1].read();
        layers[3].read();
        printPredictInput(layers[3]);
        printWeights(layers[3]);
        printWeights(layers[3], 1);
        printPredictOutput(layers[3]);
        printPredictOutput(layers[3], 1);

        // layer_schema_t* layer1Schema = layers[2].schema;
        // int osc = layer1Schema->outputHeight * layer1Schema->outputWidth;;
        // int os = layer1Schema->outputDepth * layer1Schema->outputHeight * layer1Schema->outputWidth;
        // int ws = layer1Schema->inputDepth * layer1Schema->operationHeight * layer1Schema->operationWidth + 1;
        // printf("After Run ws = %d:\n", ws);
        // printf("b = %.8lf, w = \n", (layers[2].weights)[0]);
        // printMatrixlf8(layers[2].weights + 1, layers[2].schema->operationHeight, layers[2].schema->operationWidth);
        // printMatrixlf8(layers[2].predictInput, layers[2].schema->inputHeight, layers[2].schema->inputWidth);
        // printMatrixlf8(layers[2].predictOutput, layers[2].schema->outputHeight, layers[2].schema->outputWidth);
    } else if (step == 3 && batchIndex == 0) {
        // printf("After Train:\n");
        // printMatrixlf8(layers[2].trainInput, layers[2].schema->outputHeight, layers[2].schema->outputWidth);
        // printMatrixlf8(layers[2].trainOutput, layers[2].schema->inputHeight, layers[2].schema->inputWidth);
    }
    return 0;
}

int main(int argc, const char* argv[]) {
    cudaError_t cudaStatus;
    ModelFacadeBuilder builder;
    int ret = 0;

    if (parseCliArguments(&cli, argc, argv)) {
        return -1;
    }

    if (cli.showGpus) {
        showDevices();
        return 0;
    }

    cudaStatus = cudaSetDevice(cli.device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "无法切换到设备 %d. 您的GPU是否支持CUDA?", cli.device);
        return -1;
    }

    initConfig(&config, &builder);
    if (readConfig(&config, cli.configPath)) {
        return -1;
    }

    if (builder.build(&model)) {
        return -1;
    }
    model.setStudyRate(config.studyRate);
    model.setAttenuationRate(config.attenuationRate);
    model.setRoundCount(config.roundCount);
    // model.setTrainListener(trainListener);
    // for (int i = 0; i < 8; i++) layers[i].setLayerSchema(model.layerAt(i));
    model.printSchema();

    ret = ret || trainImageSet.read(config.trainImage);
    ret = ret || trainLabelSet.read(config.trainLabel);
    ret = ret || testImageSet.read(config.testImage);
    ret = ret || testLabelSet.read(config.testLabel);
    if (ret) {
        return -1;
    }

    printMatrixu(trainLabelSet.labels, 5, 5);

    if (config.printMemoryUsed) {
        int memory = model.getTotalMemoryUsed();
        printf("Total Memory Used = %d Bytes, %.2lf MB\n", memory, memory / 1024.0 / 1024.0);
    }

    if (cli.predict && !cli.train) {
        if (strcmp(cli.loadPath, "") != 0) {
            model.loadModel(cli.loadPath);
        } else if (strcmp(config.loadPath, "") != 0) {
            model.loadModel(config.loadPath);
        } else {
            fprintf(stderr, "没有模型读取路径\n");
            return -1;
        }
    }

    if (cli.train) {
        if (strcmp(cli.savePath, "") == 0 && strcmp(config.loadPath, "") == 0) {
            fprintf(stderr, "没有模型保存路径\n");
            return -1;
        }
    }

    if (cli.train) {
        if (model.train(trainImageSet.images, trainLabelSet.labels, config.trainImageCount, config.printTrainProcess)) {
            return -1;
        }
    }

    if (cli.predict) {
        unsigned char* output = new unsigned char[config.predictImageCount];
        if (model.predict(testImageSet.images, output, config.predictImageCount)) {
            delete [] output;
            return -1;
        }

        int acc = 0;
        for (int i = 0; i < config.predictImageCount; i++) {
            if (testLabelSet.labels[i] == output[i]) {
                acc++;
            }
            if (i < config.predictOutputCount && config.printPredictOutput) {
                printf("expect %d, got %d\n", testLabelSet.labels[i], output[i]);
            }
        }
        if (config.printPredictAccuracyRate) {
            printf("准确率 = %10.6lf%%\n", acc * 100.0 / config.predictImageCount);
        }

        delete [] output;
    }

    if (cli.train) {
        if (strcmp(cli.savePath, "") != 0) {
            model.saveModel(cli.savePath);
        } else if (strcmp(config.loadPath, "") != 0) {
            model.saveModel(config.loadPath);
        } else {
            fprintf(stderr, "没有模型保存路径\n");
            return -1;
        }
    }
    return 0;
}

int showDevices() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("deviceCount = %d\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("使用GPU device %d: %s\n", i, devProp.name);
        printf("设备全局内存总量: %zu MB\n", devProp.totalGlobalMem / 1024 / 1024);
        printf("SM的数量: %d\n", devProp.multiProcessorCount);
        printf("每个线程块的共享内存大小: %zu KB\n", devProp.sharedMemPerBlock / 1024);
        printf("每个线程块的最大线程数: %d\n", devProp.maxThreadsPerBlock);
        printf("设备上一个线程块（Block）种可用的32位寄存器数量: %d\n", devProp.regsPerBlock);
        printf("每个EM的最大线程数: %d\n", devProp.maxThreadsPerMultiProcessor);
        printf("每个EM的最大线程束数: %d\n", devProp.maxThreadsPerMultiProcessor / 32);
        printf("设备上多处理器的数量: %d\n", devProp.multiProcessorCount);
        printf("======================================================\n");
    }
    return 0;
}

void initConfig(model_config_t* config, ModelFacadeBuilder* builder) {
    config->builder = builder;
    config->trainImage[0] = 0;
    config->trainLabel[0] = 0;
    config->testImage[0] = 0;
    config->testLabel[0] = 0;
    config->loadPath[0] = 0;
    config->savePath[0] = 0;
    config->memoryCount = 0;
    config->batchSize = 0;
    config->trainImageCount = 0;
    config->studyRate = 0.01;
    config->attenuationRate = 0.912;
    config->printMemoryUsed = 1;
    config->printTrainProcess = 1;
    config->printPredictOutput = 1;
    config->printPredictAccuracyRate = 1;
}

int readConfig(model_config_t* config, const char* configPath) {
    Config reader(config);
    reader.expectString(MODULE_INPUT, "trainImage", config->trainImage);
    reader.expectString(MODULE_INPUT, "trainLabel", config->trainLabel);
    reader.expectString(MODULE_INPUT, "testImage", config->testImage);
    reader.expectString(MODULE_INPUT, "testLabel", config->testLabel);
    reader.expectString(MODULE_GLOBAL, "loadPath", config->loadPath);
    reader.expectString(MODULE_GLOBAL, "savePath", config->savePath);
    reader.expectInteger(MODULE_GLOBAL, "roundCount", &(config->roundCount));
    reader.expectInteger(MODULE_GLOBAL, "memoryCount", &(config->memoryCount));
    reader.expectInteger(MODULE_GLOBAL, "batchSize", &(config->batchSize));
    reader.expectInteger(MODULE_GLOBAL, "trainImageCount", &(config->trainImageCount));
    reader.expectInteger(MODULE_GLOBAL, "predictImageCount", &(config->predictImageCount));
    reader.expectInteger(MODULE_GLOBAL, "predictOutputCount", &(config->predictOutputCount));
    reader.expectDouble(MODULE_GLOBAL, "studyRate", &(config->studyRate));
    reader.expectDouble(MODULE_GLOBAL, "attenuationRate", &(config->attenuationRate));
    reader.expectInteger(MODULE_GLOBAL, "printMemoryUsed", &(config->printMemoryUsed));
    reader.expectInteger(MODULE_GLOBAL, "printTrainProcess", &(config->printTrainProcess));
    reader.expectInteger(MODULE_GLOBAL, "printPredictOutput", &(config->printPredictOutput));
    reader.expectInteger(MODULE_GLOBAL, "printPredictAccuracyRate", &(config->printPredictAccuracyRate));
    reader.beforeModule(&beforeModule);
    reader.expectLayer(MODULE_MODEL, "Input", &readLayer);
    reader.expectLayer(MODULE_MODEL, "Dense", &readLayer);
    reader.expectLayer(MODULE_MODEL, "Convolution", &readLayer);
    reader.expectLayer(MODULE_MODEL, "Pooling", &readLayer);
    reader.expectLayer(MODULE_MODEL, "Scale", &readLayer);
    reader.expectLayer(MODULE_MODEL, "Output", &readLayer);
    return reader.read(configPath);
}

int beforeModule(void* dist, const char* module) {
    model_config_t* config = (model_config_t*) dist;
    if (strcmp(module, "model") == 0) {
        if (config->memoryCount <= 0) {
            fprintf(stderr, "内存分配个数必须为正数\n");
            return 1;
        }
        if (config->batchSize <= 0) {
            fprintf(stderr, "batchSize必须为正数\n");
            return 1;
        }
        if (config->memoryCount < config->trainImageCount) {
            fprintf(stderr, "内存分配个数必须大于等于训练图片个数\n");
            return 1;
        }
        if (config->memoryCount < config->predictImageCount) {
            fprintf(stderr, "内存分配个数必须大于等于预测图片个数\n");
            return 1;
        }
        config->builder->setMemory(config->memoryCount, config->batchSize);
    }
    return 0;
}

int readLayer(void* dist, const char* layerName, const int n, const int argv[]) {
    model_config_t* config = (model_config_t*) dist;
    if (strcmp(layerName, "Input") == 0) {
        if (n != 2) {
            fprintf(stderr, "输入层必须是两个参数, 实际上获得 %d 个参数\n", n);
            return 1;
        }
        config->builder->input(argv[0], argv[1]);
    } else if (strcmp(layerName, "Convolution") == 0) {
        if (n == 3) {
            config->builder->convolution(argv[0], argv[1], argv[2]);
        } else if (n == 4) {
            config->builder->convolution(argv[0], argv[1], argv[2], argv[3]);
        } else if (n == 5) {
            config->builder->convolution(argv[0], argv[1], argv[2], argv[3], argv[4]);
        } else if (n == 7) {
            config->builder->convolution(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
        } else if (n == 9) {
            config->builder->convolution(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8]);
        } else {
            fprintf(stderr, "卷积层参数个数应该在以下参数个数之中：3、4、7、9, 实际上获得 %d 个参数\n", n);
            return 1;
        }
    } else if (strcmp(layerName, "Pooling") == 0) {
        if (n == 1) {
            config->builder->pooling(argv[0]);
        } else if (n == 2) {
            config->builder->pooling(argv[0], argv[1]);
        } else if (n == 3) {
            config->builder->pooling(argv[0], argv[1], argv[2]);
        } else if (n == 4) {
            config->builder->pooling(argv[0], argv[1], argv[2], argv[3]);
        } else if (n == 6) {
            config->builder->pooling(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
        } else if (n == 8) {
            config->builder->pooling(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
        } else {
            fprintf(stderr, "池化层参数个数应该在以下参数个数之中：1、2、3、4、6、8, 实际上获得 %d 个参数\n", n);
            return 1;
        }
    } else if (strcmp(layerName, "Dense") == 0) {
        if (n != 1) {
            fprintf(stderr, "全连接层必须是一个参数, 实际上获得 %d 个参数\n", n);
            return 1;
        }
        config->builder->dense(argv[0]);
    } else if (strcmp(layerName, "Scale") == 0) {
        if (n != 0) {
            fprintf(stderr, "缩放层必须是零个参数, 实际上获得 %d 个参数\n", n);
            return 1;
        }
        config->builder->scale();
    } else if (strcmp(layerName, "Output") == 0) {
        if (n != 0) {
            fprintf(stderr, "输出层必须是零个参数, 实际上获得 %d 个参数\n", n);
            return 1;
        }
        config->builder->output();
    } else {
        fprintf(stderr, "未知的层类型 %s\n", layerName);
        return 1;
    }
    return 0;
}
