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

void printTrainOuput(LayerFacade& layer, int depth = 0) {
    int size = layer.schema->inputWidth * layer.schema->inputHeight;
    printMatrixlf8(layer.trainOutput + depth * size, layer.schema->inputWidth, layer.schema->inputHeight);
}

void printWeights(LayerFacade& layer, int depth = 0) {
    if (layer.schema->type == LAYER_TYPE_CONVOLUTION) {
        int size = layer.schema->inputDepth * layer.schema->operationWidth * layer.schema->operationHeight + 1;
        printf("b[%d] = %.8lf, w[%d] = \n", depth, layer.weights[depth * size], depth);
        printMatrixlf8(layer.weights + depth * size + 1, layer.schema->operationWidth, layer.schema->operationHeight);
    } else if (layer.schema->type == LAYER_TYPE_DENSE) {
        int size = layer.schema->inputDepth * layer.schema->inputWidth * layer.schema->inputHeight + 1;
        printf("b[%d] = %.8lf, w[%d] = \n", depth, layer.weights[depth * size], depth);
        printMatrixlf8(layer.weights + depth * size + 1, size - 1, 1);
    }
}

void printDweights(LayerFacade& layer, int depth = 0) {
    if (layer.schema->type == LAYER_TYPE_CONVOLUTION) {
        int size = layer.schema->inputDepth * layer.schema->operationWidth * layer.schema->operationHeight + 1;
        printf("db[%d] = %.8lf, dw[%d] = \n", depth, layer.dweights[depth * size], depth);
        printMatrixlf8(layer.dweights + depth * size + 1, layer.schema->operationWidth, layer.schema->operationHeight);
    } else if (layer.schema->type == LAYER_TYPE_DENSE) {
        int size = layer.schema->inputDepth * layer.schema->inputWidth * layer.schema->inputHeight + 1;
        printf("db[%d] = %.8lf, dw[%d] = \n", depth, layer.dweights[depth * size], depth);
        printMatrixlf8(layer.dweights + depth * size + 1, size - 1, 1);
    }
}

void printConvolutionLayer(LayerFacade& layer) {
    layer.read();
    printPredictInput(layer);
    printWeights(layer);
    printWeights(layer, 1);
    printPredictOutput(layer);
    printPredictOutput(layer, 1);
    printDweights(layer);
    printDweights(layer, 1);
}

void printOutputY(LayerFacade& layer, int batchSize) {
    layer_schema_t* schema = layer.schema;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    int offset = batchSize * 1 + batchSize * 1 + batchSize * outputSize;
    printMatrixlf8(layer.predictTemp + offset, outputSize, 1);
}

int trainListener(model_schema_t* mem, int batchIndex, int step) {
    if (step == 2) {
        printf("[After Train, Before apply]\n");
        printf("Layer 2\n");
        printConvolutionLayer(layers[2]);
        printf("Layer 4\n");
        printConvolutionLayer(layers[4]);
        // layers[1].read();
        // printPredictOutput(layers[1]);

        // layers[3].read();
        // printPredictOutput(layers[3]);

        // layers[5].read();
        // printPredictOutput(layers[5]);

        // printf("Layer 8 = \n");
        // layers[8].read();
        // printOutputY(layers[8], config.batchSize);
        // printMatrixlf8(layers[8].trainOutput, 10, 1);

        // printf("Layer 7 = \n");
        // layers[7].read();
        // printMatrixlf8(layers[7].predictInput, 500, 1);
        // printWeights(layers[7]);
        // printDweights(layers[7]);
        // printMatrixlf8(layers[7].trainOutput, 10, 1);

        // printf("Layer 6 = \n");
        // layers[6].read();
        // printMatrixlf8(layers[6].trainOutput, 10, 1);

        // printf("Layer 5 = \n");
        // layers[5].read();
        // printTrainOuput(layers[5]);

        // printf("Layer 3 = \n");
        // layers[3].read();
        // printTrainOuput(layers[3]);
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
    reader.expectLayer(MODULE_MODEL, "Relu", &readLayer);
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
    } else if (strcmp(layerName, "Relu") == 0) {
        if (n != 0) {
            fprintf(stderr, "线性整流层必须是零个参数, 实际上获得 %d 个参数\n", n);
            return 1;
        }
        config->builder->relu();
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
