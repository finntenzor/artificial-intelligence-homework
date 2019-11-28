/**
 * layer_run.cu
 */

#include "layer_run.h"
#include "cuda.h"

/*
 * ����������Ӧ�±�
 * ���еĲ�ͳһ����ά���飬�ĸ�ά��������
 * ��� ÿ��batch����batchSize����
 * ͨ���� ���Ϊ��������-1
 * �к�
 * �к�
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
        fprintf(stderr, "�� %d �㷢������CUDA��Ϣ: %s\n", layerIndex, cudaGetErrorString(cudaStatus));
        return 1;
    }
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)  {
        fprintf(stderr, "�� %d �㷢������CUDA��Ϣ: %s\n", layerIndex, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}
