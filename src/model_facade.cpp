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
        fprintf(stderr, "�޷���д��ģʽ��ģ���ļ�\n");
        return;
    }
    if (hostWeights == NULL) {
        fprintf(stderr, "ģ��Ȩ��Ϊ�գ��޷�����");
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
        fprintf(stderr, "�޷��Զ�ȡģʽ��ģ���ļ�\n");
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
        fprintf(stderr, "�ڴ治�㣬�޷�Ԥ��˹�ģ������\n");
        return -1;
    }
    if (totalCount != 0) {
        this->inputCount = totalCount;
    }

    if (this->hostWeights == NULL) {
        fprintf(stderr, "û��ģ��Ȩ�أ��޷�����Ԥ��\n");
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
        fprintf(stderr, "�ڴ治�㣬�޷�ѵ���˹�ģ������\n");
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
        fprintf(stderr, "û����һ��\n");
        return NULL;
    }
    return &schemas[index];
}

void ModelFacade::setStudyRate(double studyRate) {
    this->studyRate = studyRate;
}
