/**
 * main.cpp
 */

#include "main.h"
#include <cmath>

const int TRAIN_IMAGE_COUNT = 60000;
const int TRAIN_BATCH_SIZE = 10000;
const int PREDICT_IMAGE_COUNT = 100;

cli_arguments_t cli;
ImageSet trainImageSet;
LabelSet trainLabelSet;
ImageSet testImageSet;
LabelSet testLabelSet;
ModelFacade model;

const int layerCount = 3;
LayerFacade layerFacade[layerCount];

int showDevices();

void readAllLayer() {
    for (int i = 0; i < layerCount; i++) {
        layerFacade[i].read();
    }
}

int inspect(model_schema_t* mem, int batchIndex, int step) {
    // readAllLayer();
    int channels = TRAIN_BATCH_SIZE < 10 ? TRAIN_BATCH_SIZE : 10;
    int weightSize = 28 * 28 + 1;
    int keyx = 0;
    double* keyxWeights = layerFacade[1].weights + keyx * weightSize;
    double* keyxDweights = layerFacade[1].dweights + keyx * weightSize;
    double* keyxb = keyxWeights;
    double* keyxw = keyxWeights + 1;
    double* keyxdw = keyxDweights + 1;

    if (step == 3) {
        // layerFacade[1].read();
        // layerFacade[2].read();
        // layerFacade[3].read();
        // layerFacade[4].read();
        // layerFacade[5].read();
        // double* y = layerFacade[2].predictTemp + TRAIN_BATCH_SIZE * 10 + TRAIN_BATCH_SIZE * 1;
        // printMatrixlf4(y, 10, 10);
        // printMatrixlf4(layerFacade[2].trainOutput, 10, 10);
        // printf("b = %.8lf\n", *keyxb);
        // printf("STEP 3\n");
        // printMatrixlf(keyxw, 28, 28);
        // printMatrixlf(keyxdw, 28, 28);
        // printMatrixlf(layerFacade[3].predictInput, 10, 5);
        // printMatrixlf(layerFacade[3].trainOutput, 10, 5);
        // printMatrixlf(layerFacade[2].trainOutput, 10, 5);
        // printMatrixlf6(layerFacade[2].weights, 11, 10);
        // printMatrixlf6(layerFacade[2].dweights, 11, 10);
        // printMatrixlf6(layerFacade[2].trainInput, 10, 20);
        // printMatrixlf6(layerFacade[3].trainOutput, 10, 20);
        // printMatrixlf6(layerFacade[2].trainOutput, 64, 10);
        // printMatrixlf6(layerFacade[1].trainOutput, 10, 20);
        // for (int i = 0; i < 10; i++) {
        //     printMatrixlf6(layerFacade[2].weights + i * 65 + 1, 8, 8);
        // }
        // printMatrixlf(layerFacade[2].dweights, 11, 10);
        // printMatrixlf8(layerFacade[3].trainOutput, 10, 20);
        // printMatrixlf8(layerFacade[2].trainInput, 10, 20);
        // printMatrixlf8(layerFacade[2].trainTemp, 10, 20);
        // printMatrixlf8(layerFacade[2].trainOutput, 64, 20);

        // printMatrixlf4(layerFacade[2].trainInput, 10, 20);
        // printMatrixlf4(layerFacade[2].trainOutput, 64, 20);

        // printMatrixlf4(layerFacade[1].trainInput, 64, 20);
        // for (int i = 0; i < 2; i++) {
        //     printf("db = %.4lf, dw = \n", *(layerFacade[1].dweights + i * 785));
        //     printMatrixlf4(layerFacade[1].dweights + i * 785 + 1, 28, 28);
        //     printf("b = %.4lf, w = \n", *(layerFacade[1].weights + i * 785));
        //     printMatrixlf4(layerFacade[1].weights + i * 785 + 1, 28, 28);
        // }
    }
    // else if (step == 3) {
    //     layerFacade[1].read();
    //     printf("STEP 3\n");
    //     printf("b = %.8lf\n", *key1b);
    //     printMatrixlf(key1w, 28, 28);
    //     if (isnan(*key1b)) {
    //         layerFacade[2].read();
    //         printMatrixlf6(layerFacade[2].trainOutput, 10, 5);
    //         printMatrixlf6(layerFacade[2].predictOutput, 10, 5);
    //         printMatrixlf6(layerFacade[2].predictInput, 10, 5);
    //         return 1;
    //     }
    // }

    return 0;
}

void descriptModel(ModelFacade* model) {
    ModelFacadeBuilder builder;
    builder.setMemory(TRAIN_IMAGE_COUNT, TRAIN_BATCH_SIZE);
    builder.input(28, 28); // 0
    // builder.dense(10); // 1
    // builder.scale(); // 2
    // builder.dense(64); // 1
    // builder.scale(); // 4
    builder.dense(10); // 2
    // builder.scale(); // 4
    builder.output(); // 3
    builder.build(model);

    model->setStudyRate(0.01);
    model->setTrainListener(&inspect);
}

unsigned char output[PREDICT_IMAGE_COUNT];

int main(int argc, const char* argv[]) {
    parseCliArguments(&cli, argc, argv);
    if (cli.showGpus) {
        showDevices();
    } else {
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(cli.device);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Can not use device %d. Do you have a CUDA-capable GPU installed?", cli.device);
            return -1;
        }

        trainImageSet.read("./data/train-images.idx3-ubyte");
        trainLabelSet.read("./data/train-labels.idx1-ubyte");
        testImageSet.read("./data/t10k-images.idx3-ubyte");
        testLabelSet.read("./data/t10k-labels.idx1-ubyte");

        printMatrixu(trainLabelSet.labels, 25, 1);

        descriptModel(&model);
        for (int i = 0; i < layerCount; i++) {
            layerFacade[i].setLayerSchema(model.layerAt(i));
        }

        int memory = model.getTotalMemoryUsed();
        printf("Total Memory Used = %d Bytes, %.2lf MB\n", memory, memory / 1024.0 / 1024.0);

        if (strcmp(cli.loadPath, "") != 0) {
            model.loadModel(cli.loadPath);
        }

        if (cli.train) {
            model.train(trainImageSet.images, trainLabelSet.labels, TRAIN_IMAGE_COUNT);
        }

        if (cli.predict) {
            model.predict(testImageSet.images, output, PREDICT_IMAGE_COUNT);
            // model.predict(testImageSet.images, output, PREDICT_IMAGE_COUNT);
            // printf("׼ȷ�� = %10.6lf%%\n", model.getAccuracyRate() * 100);

            int acc = 0;
            for (int i = 0; i < PREDICT_IMAGE_COUNT; i++) {
                if (testLabelSet.labels[i] == output[i]) {
                    acc++;
                }
                if (i < 100) {
                    printf("expect %d, got %d\n", testLabelSet.labels[i], output[i]);
                }
            }
            printf("׼ȷ�� = %10.6lf%%\n", acc * 100.0 / PREDICT_IMAGE_COUNT);
        }

        if (strcmp(cli.savePath, "") != 0) {
            model.saveModel(cli.savePath);
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
        printf("ʹ��GPU device %d: %s\n", i, devProp.name);
        printf("�豸ȫ���ڴ�����: %zu MB\n", devProp.totalGlobalMem / 1024 / 1024);
        printf("SM������: %d\n", devProp.multiProcessorCount);
        printf("ÿ���߳̿�Ĺ����ڴ��С: %zu KB\n", devProp.sharedMemPerBlock / 1024);
        printf("ÿ���߳̿������߳���: %d\n", devProp.maxThreadsPerBlock);
        printf("�豸��һ���߳̿飨Block���ֿ��õ�32λ�Ĵ�������: %d\n", devProp.regsPerBlock);
        printf("ÿ��EM������߳���: %d\n", devProp.maxThreadsPerMultiProcessor);
        printf("ÿ��EM������߳�����: %d\n", devProp.maxThreadsPerMultiProcessor / 32);
        printf("�豸�϶ദ����������: %d\n", devProp.multiProcessorCount);
        printf("======================================================\n");
    }
    return 0;
}
