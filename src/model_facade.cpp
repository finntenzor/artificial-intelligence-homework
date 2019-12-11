/**
 * model_facade.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "model_facade.h"

ModelFacade::ModelFacade() {
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

int ModelFacade::saveModel(const char* filepath) {
    FILE* f = NULL;

    f = fopen(filepath, "wb");
    if (f == NULL) {
        fprintf(stderr, "无法以写入模式打开模型文件 %s\n", filepath);
        return 1;
    }
    if (hostWeights == NULL) {
        fprintf(stderr, "模型权重为空，无法保存");
        return 1;
    }

    fwrite(hostWeights, sizeof(double), getWeightsSize(), f);

    if (f != NULL) {
        fclose(f);
    }

    return 0;
}

int ModelFacade::loadModel(const char* filepath) {
    FILE* f = NULL;

    f = fopen(filepath, "rb");
    if (f == NULL) {
        fprintf(stderr, "无法以读取模式打开模型文件 %s\n", filepath);
        return 1;
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

    return 0;
}

void ModelFacade::randomGenerateArgs() {
    double* w;
    if (this->hostWeights != NULL) {
        delete [] this->hostWeights;
    }
    this->hostWeights = w = new double[getWeightsSize()];
    srand(time(NULL));
    for (int k = 0; k < schemaCount; k++) {
        layer_schema_t* schema = &schemas[k];
        int m = schema->weightsSize;
        double q;
        if (schema->type == LAYER_TYPE_CONVOLUTION) {
            int ws = schema->inputDepth * schema->operationWidth * schema->operationHeight + 1;
            q = sqrt(3.0) / (ws - 1);
            for (int i = 0; i < schema->outputDepth; i++) {
                w[i * ws] = 0;
                for (int j = 1; j < ws; j++) {
                    w[i * ws + j] = ((rand() % 1024) / 1024.0 - 0.5) * q;
                }
            }
        } else {
            q = sqrt(3.0) / schema->predictInputSize;
            for (int i = 0; i < m; i++) {
                w[i] = ((rand() % 1024) / 1024.0 - 0.5) * q;
            }
        }
        w += m;
    }
}

int ModelFacade::predict(unsigned char* input, unsigned char* output) {
    int ret = 0;

    if (this->hostWeights == NULL) {
        fprintf(stderr, "没有模型权重，无法进行预测\n");
        return -1;
    }

    ret = ret || modelCopyPredictInput(this, input);
    ret = ret || modelCopyToWeights(this, hostWeights);
    ret = ret || modelPredict(this);
    ret = ret || modelCopyPredictOutput(this, output);
    ret = ret || modelCopyFromWeights(this, hostWeights);
    return ret;
}

double ModelFacade::getAccuracyRate() {
    return accuracyRate;
}

int ModelFacade::train(unsigned char* trainInput, unsigned char* trainLabels, unsigned char* testInput, unsigned char* testLabels) {
    int ret = 0;

    if (this->hostWeights == NULL) {
        this->randomGenerateArgs();
    }

    ret = ret || modelCopyTrainInput(this, trainInput, testInput);
    ret = ret || modelCopyTrainLabels(this, trainLabels, testLabels);
    ret = ret || modelCopyToWeights(this, hostWeights);
    ret = ret || modelTrain(this, batchCallback);
    ret = ret || modelCopyFromWeights(this, hostWeights);
    return ret;
}

void ModelFacade::setTrainListener(int (*trainListener)(model_schema_t* mem, int batchIndex, int step)) {
    batchCallback = trainListener;
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

void ModelFacade::setAttenuationRate(double attenuationRate) {
    this->attenuationRate = attenuationRate;
}

void ModelFacade::setEpoch(int epoch) {
    this->epoch = epoch;
}

void ModelFacade::setPrintTrainProcess(int printTrainProcess) {
    this->printTrainProcess = printTrainProcess;
}

void ModelFacade::setLossCheckCount(int lossCheckCount) {
    if (lossCheckCount < 2) lossCheckCount = 2;
    this->lossCheckCount = lossCheckCount;
}

void ModelFacade::printSchema() {
    for (int i = 0; i < schemaCount; i++) {
        layer_schema_t* schema = &schemas[i];
        printf("[%d](%d, %d, %d) => (%d, %d, %d)\n", i,
            schema->inputDepth, schema->inputHeight, schema->inputWidth,
            schema->outputDepth, schema->outputHeight, schema->outputWidth
        );
    }
}
