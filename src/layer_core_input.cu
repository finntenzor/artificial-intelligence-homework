/**
 * layer_core_input.cu
 */

#include "layer_run.h"
#include "cuda.h"

__global__ void layerDevPredictInput(layer_schema_t schema, unsigned char* input) {
    // 输入层应该输入和输出宽高相等，这里不做检查
    // 输入通道数应该总是1，这里不做检查
    double* output = schema.predictOutput;
    int inputIndex = layerGetInputIndex(&schema, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    int outputIndex = layerGetCurrentOutputIndex(&schema);
    output[outputIndex] = (input[inputIndex] + 1) / 256.0;

    // DEBUG
    // printf("&output = %p, blockIdx.x = %d, blockIdx.y = %d, threadIdx.x = %d, threadIdx.y = %d, inputIndex = %d, ouputIndex = %d, input = %d, output = %lf\n", output, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, inputIndex, outputIndex, input[inputIndex], output[outputIndex]);
}

int layerPredictInput(layer_schema_t* schema, int batchSize, unsigned char* input) {
    dim3 gridSize(batchSize, schema->outputDepth);
    dim3 blockSize(schema->outputHeight, schema->outputWidth);
    layerDevPredictInput<<<gridSize, blockSize>>>(*schema, input);
    return layerIfError(schema->layerIndex);
}
