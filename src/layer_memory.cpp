/**
 * layer_memory.cpp
 */

#include "layer_memory.h"
#include "cuda.h"

int layerCopyToDevice(layer_schema_t* schema, void* dist, void* src, size_t size, const char* name) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaMemcpy(dist, src, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "�޷��� %d ��Ľ� %s ���Ƶ��Դ�, CUDA��Ϣ��%s\n", schema->layerIndex, name, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}

int layerCopyFromDevice(layer_schema_t* schema, void* dist, void* src, size_t size, const char* name) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaMemcpy(dist, src, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "�޷����Դ渴�Ƶ� %d ��� %s ���ڴ�, CUDA��Ϣ��%s\n", schema->layerIndex, name, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}

int layerCopyFromDeviceDouble(layer_schema_t* schema, void* dist, void* src, int size, const char* name) {
    return layerCopyFromDevice(schema, dist, src, size * sizeof(double), name);
}

int layerCopyPredictInput(layer_schema_t* schema, double* input) {
    return layerCopyFromDeviceDouble(schema, input, schema->predictInput, schema->predictInputSize, "Ԥ������");
}

int layerCopyPredictOutput(layer_schema_t* schema, double* output) {
    return layerCopyFromDeviceDouble(schema, output, schema->predictOutput, schema->predictOutputSize, "Ԥ�����");
}

int layerCopyPredictTemp(layer_schema_t* schema, double* temp) {
    return layerCopyFromDeviceDouble(schema, temp, schema->predictTemp, schema->predictTempSize, "Ԥ���м����");
}

int layerCopyTrainInput(layer_schema_t* schema, double* input) {
    return layerCopyFromDeviceDouble(schema, input, schema->trainInput, schema->trainInputSize, "ѵ������");
}

int layerCopyTrainOutput(layer_schema_t* schema, double* output) {
    return layerCopyFromDeviceDouble(schema, output, schema->trainOutput, schema->trainOutputSize, "ѵ�����");
}

int layerCopyTrainTemp(layer_schema_t* schema, double* temp) {
    return layerCopyFromDeviceDouble(schema, temp, schema->trainTemp, schema->trainTempSize, "ѵ���м����");
}

int layerCopyWeights(layer_schema_t* schema, double* weights) {
    return layerCopyFromDeviceDouble(schema, weights, schema->weights, schema->weightsSize, "ģ��Ȩ��");
}

int layerCopyDweights(layer_schema_t* schema, double* dweights) {
    return layerCopyFromDeviceDouble(schema, dweights, schema->dweights, schema->dweightsSize, "ģ��Ȩ�ر仯��");
}
