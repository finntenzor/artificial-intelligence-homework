/**
 * layer_core_convolution.cu
 */

#include "layer_run.h"
#include "cuda.h"

// 卷积核下标，最大为输出深度-1
__device__ int layerGetConvolutionArgsIndex(layer_schema_t* schema, int kernelIndex) {
    return kernelIndex * (schema->inputDepth * schema->operationHeight * schema->operationWidth + 1);
}

// 共有卷积核组权重和偏置
__device__ int layerGetConvolutionWeightIndexOffset(layer_schema_t* schema, int channelIndex, int rowIndex, int colIndex) {
    // 多加1是因为这个位置是偏置
    return 1 + (channelIndex * schema->operationHeight + rowIndex) * schema->operationWidth + colIndex;
}

__global__ void layerDevPredictConvolution(layer_schema_t schema) {
    double* output = schema.predictOutput;
    double* input = schema.predictInput;
    double* args = schema.weights;
    // 参数总个数 = outputDepth * (operationWidth * operationHeight * inputDepth + 1)
    int outputIndex = layerGetCurrentOutputIndex(&schema);
    int argsBasis = layerGetConvolutionArgsIndex(&schema, blockIdx.y);

    int kernelWidth = (int)schema.operationWidth;
    int kernelHeight = (int)schema.operationHeight;
    int rowBegin = threadIdx.x * schema.operationRowStep + schema.operationRowBasis;
    int colBegin = threadIdx.y * schema.operationColStep + schema.operationColBasis;

    double z = args[argsBasis]; // 偏置
    for (int k = 0; k < schema.inputDepth; k++) { // 输入通道数
        for (int i = 0; i < kernelHeight; i++) { // 输入行号偏移量
            for (int j = 0; j < kernelWidth; j++) { // 输入列号偏移量
                int row = rowBegin + i; // 输入行号
                int col = colBegin + j; // 输入列号
                int inputIndex = layerGetInputIndex(&schema, blockIdx.x, k, row, col);
                int weightIndex = argsBasis + layerGetConvolutionWeightIndexOffset(&schema, k, i, j);
                if (row < 0 || row >= schema.inputHeight || col < 0 || col >= schema.inputWidth) {
                    continue;
                }
                z += args[weightIndex] * input[inputIndex];
            }
        }
    }
    // relu激活函数
    output[outputIndex] = (z > 0 ? z : 0);
}

int layerPredictConvolution(layer_schema_t* schema, int batchSize) {
    dim3 gridSize(batchSize, schema->outputDepth);
    dim3 blockSize(schema->outputHeight, schema->outputWidth);
    layerDevPredictConvolution<<<gridSize, blockSize>>>(*schema);
    return layerIfError(schema->layerIndex);
}
