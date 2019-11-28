/**
 * layer_run.cu
 */

#include "layer_run.h"
#include "cuda.h"

/*
 * 返回输入层对应下标
 * 所有的层统一按四维数组，四个维度依次是
 * 块号 每个batch包含batchSize个块
 * 通道号 最大为输入层深度-1
 * 行号
 * 列号
 */
__device__ int layerGetInputIndex(layer_schema_t* schema, int blockIndex, int channelIndex, int rowIndex, int colIndex) {
    return ((blockIndex * schema->inputDepth + channelIndex) * schema->inputHeight + rowIndex) * schema->inputWidth + colIndex;
}

__device__ int layerGetOutputIndex(layer_schema_t* schema, int blockIndex, int channelIndex, int rowIndex, int colIndex) {
    return ((blockIndex * schema->outputDepth + channelIndex) * schema->outputHeight + rowIndex) * schema->outputWidth + colIndex;
}

__device__ int layerGetCurrentOutputIndex(layer_schema_t* schema) {
    return layerGetOutputIndex(schema, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int layerIfError(int layerIndex) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "第 %d 层发生错误，CUDA信息: %s\n", layerIndex, cudaGetErrorString(cudaStatus));
        return 1;
    }
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)  {
        fprintf(stderr, "第 %d 层发生错误，CUDA信息: %s\n", layerIndex, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}
