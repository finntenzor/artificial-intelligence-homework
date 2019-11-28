/**
 * layer_facade.cpp
 */

#include <cstdio>

#include "layer_facade.h"

void LayerFacade::freeHeap() {
    if (predictInput != NULL) {
        delete [] predictInput;
        predictInput = NULL;
    }
    if (predictOutput != NULL) {
        delete [] predictOutput;
        predictOutput = NULL;
    }
    if (predictTemp != NULL) {
        delete [] predictTemp;
        predictTemp = NULL;
    }
    if (trainInput != NULL) {
        delete [] trainInput;
        trainInput = NULL;
    }
    if (trainOutput != NULL) {
        delete [] trainOutput;
        trainOutput = NULL;
    }
    if (trainTemp != NULL) {
        delete [] trainTemp;
        trainTemp = NULL;
    }
    if (weights != NULL) {
        delete [] weights;
        weights = NULL;
    }
    if (dweights != NULL) {
        delete [] dweights;
        dweights = NULL;
    }
}

void LayerFacade::allocHeap(layer_schema_t* schema) {
    predictInput = new double[schema->predictInputSize];
    predictOutput = new double[schema->predictOutputSize];
    predictTemp = new double[schema->predictTempSize];
    trainInput = new double[schema->trainInputSize];
    trainOutput = new double[schema->trainOutputSize];
    trainTemp = new double[schema->trainTempSize];
    weights = new double[schema->weightsSize];
    dweights = new double[schema->dweightsSize];
}

LayerFacade::LayerFacade() {
    this->schema = NULL;
    predictInput = NULL;
    predictOutput = NULL;
    predictTemp = NULL;
    trainInput = NULL;
    trainOutput = NULL;
    trainTemp = NULL;
    weights = NULL;
    dweights = NULL;
}

LayerFacade::LayerFacade(layer_schema_t* schema) {
    this->schema = schema;
    allocHeap(schema);
}

LayerFacade::~LayerFacade() {
    freeHeap();
    this->schema = NULL;
}

void LayerFacade::setLayerSchema(layer_schema_t* schema) {
    freeHeap();
    this->schema = schema;
    allocHeap(schema);
}

void LayerFacade::read() {
    layerCopyPredictInput(schema, predictInput);
    layerCopyPredictOutput(schema, predictOutput);
    layerCopyPredictTemp(schema, predictTemp);
    layerCopyTrainInput(schema, trainInput);
    layerCopyTrainOutput(schema, trainOutput);
    layerCopyTrainTemp(schema, trainTemp);
    layerCopyWeights(schema, weights);
    layerCopyDweights(schema, dweights);
}

void LayerFacade::printGeneralOutputImage(int blockIndex, int depth) {
    int imageSize = schema->outputHeight * schema->outputWidth;
    printImagelf(predictOutput + (blockIndex * schema->outputDepth + depth), schema->outputWidth, schema->outputHeight);
}

void LayerFacade::printGeneralOutputMatrix(int blockIndex, int depth) {
    int matrixSize = schema->outputHeight * schema->outputWidth;
    printMatrixlf(predictOutput + (blockIndex * schema->outputDepth + depth), schema->outputWidth, schema->outputHeight);
}

void LayerFacade::printInputOutputImage(int index) {
    int imageSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    printImagelf(predictOutput + imageSize * index, schema->outputWidth, schema->outputHeight);
}

void LayerFacade::printFullConnectedArgs(int featureIndex, int depth) {
    int blockSize = (schema->inputDepth * schema->inputHeight * schema->inputWidth + 1);
    printf("b = %.2lf\n", weights[blockSize * featureIndex]);
    printMatrixlf(weights + featureIndex * blockSize + 1, schema->inputWidth, schema->inputHeight);
}

void LayerFacade::printFullConnectedOutput(int beginBlock, int endBlock = -1) {
    int blockSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    if (endBlock < 0) endBlock = beginBlock;
    for (int i = beginBlock; i <= endBlock; i++) {
        for (int j = 0; j < blockSize; j++) {
            printf("%.2lf ", predictOutput[i * blockSize + j]);
        }
        printf("\n");
    }
    printf("\n");
}
