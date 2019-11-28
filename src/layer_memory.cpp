/**
 * layer_memory.cpp
 */

#include "layer_memory.h"
#include "cuda.h"

int layerCopyToDevice(layer_schema_t* schema, void* dist, void* src, size_t size, const char* name) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaMemcpy(dist, src, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "无法第 %d 层的将 %s 复制到显存, CUDA信息：%s\n", schema->layerIndex, name, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}

int layerCopyFromDevice(layer_schema_t* schema, void* dist, void* src, size_t size, const char* name) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaMemcpy(dist, src, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "无法从显存复制第 %d 层的 %s 到内存, CUDA信息：%s\n", schema->layerIndex, name, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}

int layerCopyFromDeviceDouble(layer_schema_t* schema, void* dist, void* src, int size, const char* name) {
    return layerCopyFromDevice(schema, dist, src, size * sizeof(double), name);
}

int layerCopyPredictInput(layer_schema_t* schema, double* input) {
    return layerCopyFromDeviceDouble(schema, input, schema->predictInput, schema->predictInputSize, "预测输入");
}

int layerCopyPredictOutput(layer_schema_t* schema, double* output) {
    return layerCopyFromDeviceDouble(schema, output, schema->predictOutput, schema->predictOutputSize, "预测输出");
}

int layerCopyPredictTemp(layer_schema_t* schema, double* temp) {
    return layerCopyFromDeviceDouble(schema, temp, schema->predictTemp, schema->predictTempSize, "预测中间变量");
}

int layerCopyTrainInput(layer_schema_t* schema, double* input) {
    return layerCopyFromDeviceDouble(schema, input, schema->trainInput, schema->trainInputSize, "训练输入");
}

int layerCopyTrainOutput(layer_schema_t* schema, double* output) {
    return layerCopyFromDeviceDouble(schema, output, schema->trainOutput, schema->trainOutputSize, "训练输出");
}

int layerCopyTrainTemp(layer_schema_t* schema, double* temp) {
    return layerCopyFromDeviceDouble(schema, temp, schema->trainTemp, schema->trainTempSize, "训练中间变量");
}

int layerCopyWeights(layer_schema_t* schema, double* weights) {
    return layerCopyFromDeviceDouble(schema, weights, schema->weights, schema->weightsSize, "模型权重");
}

int layerCopyDweights(layer_schema_t* schema, double* dweights) {
    return layerCopyFromDeviceDouble(schema, dweights, schema->dweights, schema->dweightsSize, "模型权重变化量");
}
