/**
 * model_facade.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "model_facade.h"

ModelFacade::ModelFacade() {
    maxInputCount = 0;
    schemas = NULL;
    hostWeights = NULL;
    accuracyRate = 0;
    batchCallback = NULL;
}

ModelFacade::~ModelFacade() {
    if (schemas != NULL) {
        delete [] schemas;
    }
    if (hostWeights != NULL) {
        delete [] hostWeights;
    }
    batchCallback = NULL;
}

int ModelFacade::getWeightsSize() {
    return modelGetWeightsSize(this);
}

int ModelFacade::getTotalMemoryUsed() {
    return modelGetTotalMemoryUsed(this);
}

void ModelFacade::saveModel(const char* filepath) {
    FILE* f = NULL;

    f = fopen(filepath, "wb");
    if (f == NULL) {
        fprintf(stderr, "无法以写入模式打开模型文件\n");
        return;
    }
    if (hostWeights == NULL) {
        fprintf(stderr, "模型权重为空，无法保存");
        return;
    }

    fwrite(hostWeights, sizeof(double), getWeightsSize(), f);

    if (f != NULL) {
        fclose(f);
    }
}

void ModelFacade::loadModel(const char* filepath) {
    FILE* f = NULL;

    f = fopen(filepath, "rb");
    if (f == NULL) {
        fprintf(stderr, "无法以读取模式打开模型文件\n");
        return;
    }

    int n = getWeightsSize();
    if (this->hostWeights != NULL) {
        delete [] hostWeights;
    }
    this->hostWeights = new double[n];
    fread(hostWeights, sizeof(double), n, f);

    if (f != NULL) {
        fclose(f);
    }
}

void ModelFacade::randomGenerateArgs() {
    int n = getWeightsSize();
    if (this->hostWeights != NULL) {
        delete this->hostWeights;
    }
    this->hostWeights = new double[n];
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        this->hostWeights[i] = (rand() % 100) / 100.0 / 784.0;
    }
}

int ModelFacade::predict(unsigned char* input, unsigned char* output, int totalCount) {
    int ret = 0;

    if (totalCount > maxInputCount) {
        fprintf(stderr, "内存不足，无法预测此规模的数据\n");
        return -1;
    }
    if (totalCount != 0) {
        this->inputCount = totalCount;
    }

    if (this->hostWeights == NULL) {
        fprintf(stderr, "没有模型权重，无法进行预测\n");
        return -1;
    }

    ret = ret || modelCopyInput(this, input);
    ret = ret || modelCopyToWeights(this, hostWeights);
    ret = ret || modelPredict(this);
    ret = ret || modelCopyOutput(this, output);
    ret = ret || modelCopyFromWeights(this, hostWeights);
    return ret;
}

double ModelFacade::getAccuracyRate() {
    return accuracyRate;
}

int ModelFacade::train(unsigned char* input, unsigned char* labels, int totalCount) {
    int ret = 0;

    if (totalCount > maxInputCount) {
        fprintf(stderr, "内存不足，无法训练此规模的数据\n");
        return -1;
    }
    if (totalCount != 0) {
        this->inputCount = totalCount;
    }

    if (this->hostWeights == NULL) {
        this->randomGenerateArgs();
    }

    ret = ret || modelCopyInput(this, input);
    ret = ret || modelCopyLabels(this, labels);
    ret = ret || modelCopyToWeights(this, hostWeights);
    ret = ret || modelTrain(this, batchCallback);
    ret = ret || modelCopyFromWeights(this, hostWeights);
    return ret;
}

void ModelFacade::setTrainListener(int (*trainListenr)(model_schema_t* mem, int batchIndex, int step)) {
    batchCallback = trainListenr;
}

layer_schema_t* ModelFacade::layerAt(int index) {
    if (index < 0 || index > schemaCount) {
        fprintf(stderr, "没有这一层\n");
        return NULL;
    }
    return &schemas[index];
}

void ModelFacade::setStudyRate(double studyRate) {
    this->studyRate = studyRate;
}
