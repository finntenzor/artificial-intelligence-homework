/**
 * layer_core_relu.cu
 */

#include "layer_run.h"
#include "cuda.h"

__global__ void layerDevPredictRelu(double* output, double* input) {
    int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;

    double z = input[index];
    if (z < 0) {
        z *= 0.01;
    }
    output[index] = z;
}

int layerPredictRelu(layer_schema_t* schema, int batchSize) {
    dim3 gridSize(batchSize, schema->outputDepth);
    dim3 blockSize(schema->outputHeight * schema->outputWidth);
    layerDevPredictRelu<<<gridSize, blockSize>>>(schema->predictOutput, schema->predictInput);
    return layerIfError(schema->layerIndex);
}

// 求relu的导数
__global__ void layerDevTrainRelu(double* trainOutput, double* trainInput, double* input) {
    int index = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;

    double x = input[index];
    double z = x > 0 ? 1 : 0.01;
    trainOutput[index] = trainInput[index] * z;
}

int layerTrainRelu(layer_schema_t* schema, int batchSize) {
    dim3 gridSize(batchSize, schema->outputDepth);
    dim3 blockSize(schema->outputHeight * schema->outputWidth);
    layerDevTrainRelu<<<gridSize, blockSize>>>(schema->trainOutput, schema->trainInput, schema->predictInput);
    return layerIfError(schema->layerIndex);
}
