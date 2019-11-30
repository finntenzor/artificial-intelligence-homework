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

    builder.build(&model);
    model.setStudyRate(config.studyRate);
    model.setAttenuationRate(config.attenuationRate);
    model.setRoundCount(config.roundCount);

    ret = ret || trainImageSet.read(config.trainImage);
    ret = ret || trainLabelSet.read(config.trainLabel);
    ret = ret || testImageSet.read(config.testImage);
    ret = ret || testLabelSet.read(config.testLabel);
    if (ret) {
        return -1;
    }

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
        }
        config->builder->input(argv[0], argv[1]);
    } else if (strcmp(layerName, "Dense") == 0) {
        if (n != 1) {
            fprintf(stderr, "全连接层必须是一个参数, 实际上获得 %d 个参数\n", n);
        }
        config->builder->dense(argv[0]);
    } else if (strcmp(layerName, "Output") == 0) {
        if (n != 0) {
            fprintf(stderr, "输出层必须是零个参数, 实际上获得 %d 个参数\n", n);
        }
        config->builder->output();
    } else {
        fprintf(stderr, "未知的层类型 %s\n", layerName);
    }
    return 0;
}
