/**
 * layer_core_tanh.cu
 */

#include "layer_run.h"
#include "cuda.h"

__global__ void layerDevPredictTanh(double* output, double* input) {
    int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;

    double z = input[index];
    output[index] = tanh(z);
}

int layerPredictTanh(layer_schema_t* schema, int batchSize) {
    dim3 gridSize(batchSize, schema->outputDepth);
    dim3 blockSize(schema->outputHeight * schema->outputWidth);
    layerDevPredictTanh<<<gridSize, blockSize>>>(schema->predictOutput, schema->predictInput);
    return layerIfError(schema->layerIndex);
}

// 求tanh的导数
__global__ void layerDevTrainTanh(double* trainOutput, double* trainInput) {
    int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;

    double y = trainInput[index];
    trainOutput[index] = 1 - y * y;
}

int layerTrainTanh(layer_schema_t* schema, int batchSize) {
    dim3 gridSize(batchSize, schema->outputDepth);
    dim3 blockSize(schema->outputHeight * schema->outputWidth);
    layerDevTrainTanh<<<gridSize, blockSize>>>(schema->trainOutput, schema->trainInput);
    return layerIfError(schema->layerIndex);
}
