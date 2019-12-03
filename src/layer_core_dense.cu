/**
 * layer_core_dense.cu
 */

#include "layer_run.h"
#include "cuda.h"

// Ȩ�����±꣬���Ϊ������-1
__device__ int layerGetDenseArgsIndex(layer_schema_t* schema, int depthIndex) {
    return depthIndex * (schema->inputDepth * schema->inputHeight * schema->inputWidth + 1);
}

__global__ void layerDevPredictDense(double* output, double* input, double* weights, int inputSize) {
    int outputIndex = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
    int weightBeginIndex = (blockIdx.y * blockDim.x + threadIdx.x) * (inputSize + 1);
    int inputBeginIndex = blockIdx.x * inputSize;

    // printf("blockIdx.x = %d, blockIdx.y = %d, threadIdx.x = %d, inputSize = %d, outputIndex = %d, weightBeginIndex = %d, inputBeginIndex = %d\n",
    //     blockIdx.x, blockIdx.y, threadIdx.x, inputSize, outputIndex, weightBeginIndex, inputBeginIndex
    // );

    double z = weights[weightBeginIndex]; // ƫ��b
    for (int i = 0; i < inputSize; i++) {
        int weightIndex = weightBeginIndex + 1 + i; // Ȩ��w
        z += weights[weightIndex] * input[inputBeginIndex + i];
    }
    output[outputIndex] = z;
}

int layerPredictDense(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    dim3 gridSize(batchSize, schema->outputDepth);
    dim3 blockSize(schema->outputHeight * schema->outputWidth);
    layerDevPredictDense<<<gridSize, blockSize>>>(schema->predictOutput, schema->predictInput, schema->weights, inputSize);
    return layerIfError(schema->layerIndex);
}

// ����ǰһ��ĵ���
__global__ void layerDevTrainDense1(double* trainOutput, double* weights, double* trainInput, int outputSize, int weightsSize) {
    int xIndex = blockIdx.y * blockDim.x + threadIdx.x;
    int trainOutputIndex = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
    int trainInputBeginIndex = blockIdx.x * outputSize;
    double value = 0;
    for (int i = 0; i < outputSize; i++) {
        // i��ͨ����
        int weightIndex = i * weightsSize// ��i��ͨ����Ȩ�ز���ҳ
            + 1 + xIndex; // ������x�����

        // ��һ���ͨ�������Ͷ�Ӧλ��Ȩ�س˻�֮��
        // �������Ϊ��ʧ������ʾΪ����ʧ֮��
        value += trainInput[trainInputBeginIndex + i] * weights[weightIndex];
    }
    trainOutput[trainOutputIndex] = value;
}

int layerTrainDense1(layer_schema_t* schema, int batchSize) {
    int weightsSize = schema->inputDepth * schema->inputHeight * schema->inputWidth + 1;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize, schema->inputDepth);
    dim3 blockSize(schema->inputHeight * schema->inputWidth);
    layerDevTrainDense1<<<gridSize, blockSize>>>(schema->trainOutput, schema->weights, schema->trainInput, outputSize, weightsSize);
    return layerIfError(schema->layerIndex);
}

// ����w�仯��
__global__ void layerDevTrainDense2(double* dweights, double* predictInput, double* trainInput, int batchSize, int inputSize, int outputSize) {
    int xIndex = blockIdx.y * blockDim.x + threadIdx.x;
    int dweightsIndex = blockIdx.x * (inputSize + 1) // ����ҳ�±���ʼλ��
        + 1 + xIndex; // 1��ƫ�� �ټ�w��λ��
    double value = 0;
    for (int i = 0; i < batchSize; i++) {
        // i�ǿ�id��ÿһ��w��һ��������Ȼֻ����һ��
        int trainInputIndex = i * outputSize // ��i��block��Ӧ�����ҳ
            + blockIdx.x; // ֻȡ��ǰͨ���ĵ���
        value += trainInput[trainInputIndex] * predictInput[i * inputSize + xIndex]; // ����w��Ӧλ�õ�x
    }
    dweights[dweightsIndex] = value / batchSize;
}

int layerTrainDense2(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(outputSize, schema->inputDepth);
    dim3 blockSize(schema->inputHeight * schema->inputWidth);
    layerDevTrainDense2<<<gridSize, blockSize>>>(schema->dweights, schema->predictInput, schema->trainInput, batchSize, inputSize, outputSize);
    return layerIfError(schema->layerIndex);
}

// ����b�仯��
__global__ void layerDevTrainDense3(double* dweights, double* trainInput, int batchSize, int inputSize, int outputSize) {
    int dweightsIndex = blockIdx.x * (inputSize + 1); // ����ҳ�±���ʼλ�� �±���ƫ�ÿ�ʼ����0
    double value = 0;
    for (int i = 0; i < batchSize; i++) {
        // i�ǿ�id��ÿһ��b��һ��������Ȼֻ����һ��
        int trainInputIndex = i * outputSize // ��i��block��Ӧ�����ҳ
            + blockIdx.x; // ֻȡ��ǰͨ���ĵ���
        value += trainInput[trainInputIndex]; // ����b�ǳ��� ��b�ĵ�������1 ֱ�Ӱ���һ�㵼����ͼ���
    }
    dweights[dweightsIndex] = value / batchSize;
}

int layerTrainDense3(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(outputSize); // ͨ����
    dim3 blockSize(1);
    layerDevTrainDense3<<<gridSize, blockSize>>>(schema->dweights, schema->trainInput, batchSize, inputSize, outputSize);
    return layerIfError(schema->layerIndex);
}

int layerTrainDense(layer_schema_t* schema, int batchSize) {
    int ret = 0;
    ret = ret || layerTrainDense1(schema, batchSize);
    ret = ret || layerTrainDense2(schema, batchSize);
    ret = ret || layerTrainDense3(schema, batchSize);
    return ret;
}
