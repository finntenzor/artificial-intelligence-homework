/**
 * main_debug.cpp
 */

#include "main_debug.h"


void printGeneralOutputImage(LayerFacade& layer, int blockIndex, int depth) {
    layer_schema_t* schema = layer.schema;
    int imageSize = schema->outputHeight * schema->outputWidth;
    printImagelf(layer.predictOutput + (blockIndex * schema->outputDepth + depth), schema->outputWidth, schema->outputHeight);
}

void printGeneralOutputMatrix(LayerFacade& layer, int blockIndex, int depth) {
    layer_schema_t* schema = layer.schema;
    int matrixSize = schema->outputHeight * schema->outputWidth;
    printMatrixlf(layer.predictOutput + (blockIndex * schema->outputDepth + depth), schema->outputWidth, schema->outputHeight);
}

void printInputOutputImage(LayerFacade& layer, int index) {
    layer_schema_t* schema = layer.schema;
    int imageSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    printImagelf(layer.predictOutput + imageSize * index, schema->outputWidth, schema->outputHeight);
}

void printFullConnectedArgs(LayerFacade& layer, int featureIndex, int depth) {
    layer_schema_t* schema = layer.schema;
    int blockSize = (schema->inputDepth * schema->inputHeight * schema->inputWidth + 1);
    printf("b = %.2lf\n", layer.weights[blockSize * featureIndex]);
    printMatrixlf(layer.weights + featureIndex * blockSize + 1, schema->inputWidth, schema->inputHeight);
}

void printFullConnectedOutput(LayerFacade& layer, int beginBlock, int endBlock = -1) {
    layer_schema_t* schema = layer.schema;
    int blockSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    if (endBlock < 0) endBlock = beginBlock;
    for (int i = beginBlock; i <= endBlock; i++) {
        for (int j = 0; j < blockSize; j++) {
            printf("%.2lf ", layer.predictOutput[i * blockSize + j]);
        }
        printf("\n");
    }
    printf("\n");
}


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

void print3DArray(double* dist, int width, int height, int depth) {
    for (int k = 0; k < depth; k++) {
        printf("----------%d----------\n", k);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int index = (k * height + i) * width + j;
                printf("%.8lf ", dist[index]);
            }
            printf("\n");
        }
    }
}

void printWholeConvolutionLayer(LayerFacade& layer, int m = 0) {
    layer.read();
    layer_schema_t* schema = layer.schema;
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int weightsSize = schema->inputDepth * schema->operationWidth * schema->operationWidth + 1;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    printf("PredictInput:\n");
    print3DArray(layer.predictInput + m * inputSize, schema->inputWidth, schema->inputHeight, schema->inputDepth);
    for (int i = 0; i < schema->outputDepth; i++) {
        double* kernel = layer.weights + weightsSize * i;
        printf("Kernel %d, b = %.8lf, w = \n", i, kernel[0]);
        print3DArray(kernel + 1, schema->operationWidth, schema->operationHeight, schema->inputDepth);
    }
    printf("PredictOutput:\n");
    print3DArray(layer.predictOutput + m * outputSize, schema->outputWidth, schema->outputHeight, schema->outputDepth);
    printf("TrainInput:\n");
    print3DArray(layer.trainInput + m * outputSize, schema->outputWidth, schema->outputHeight, schema->outputDepth);
    for (int i = 0; i < schema->outputDepth; i++) {
        double* kernel = layer.dweights + weightsSize * i;
        printf("Kernel %d, db = %.8lf, dw = \n", i, kernel[0]);
        print3DArray(kernel + 1, schema->operationWidth, schema->operationHeight, schema->inputDepth);
    }
    printf("TrainOutput:\n");
    print3DArray(layer.trainOutput + m * inputSize, schema->inputWidth, schema->inputHeight, schema->inputDepth);
}

void printOutputY(LayerFacade& layer, int batchSize) {
    layer_schema_t* schema = layer.schema;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    int offset = batchSize * 1 + batchSize * 1 + batchSize * outputSize;
    printMatrixlf8(layer.predictTemp + offset, outputSize, 1);
}
