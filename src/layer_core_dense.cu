/**
 * layer_core_dense.cu
 */

#include "layer_run.h"
#include "cuda.h"

// Ȩ�����±꣬���Ϊ������-1
__device__ int layerGetDenseArgsIndex(layer_schema_t* schema, int depthIndex) {
    return depthIndex * (schema->inputDepth * schema->inputHeight * schema->inputWidth + 1);
}

__global__ void layerDevPredictDense(double* output, double* input, double* weights, int inputSize, int outputSize) {
    int outputIndex = blockIdx.x * outputSize + threadIdx.x;
    int weightBeginIndex = threadIdx.x * (inputSize + 1);

    // DEBUG
    // printf("block = %d, outputIndex = %d, argsBasis = %d, weightsSize = %d\n", blockIdx.x, outputIndex, argsBasis, weightsSize);

    // double z = 0; // ƫ��b
    double z = weights[weightBeginIndex]; // ƫ��b
    for (int i = 0; i < inputSize; i++) {
        int weightIndex = weightBeginIndex + 1 + i; // Ȩ��w
        z += weights[weightIndex] * input[blockIdx.x * inputSize + i];

        // DEBUG
        // if (outputIndex == 74) {
        //     printf("%d: w = %lf, x = %lf, z = %lf\n", i, args[weightIndex], input[i], z);
        // }
    }
    // ������Ԫ���������й�һ��
    // z /= sqrt((double)(inputSize + 1));
    z /= (inputSize / 2);
    // relu�����
    output[outputIndex] = (z > 0 ? z : 0);
    // printf("output[%d] = %lf (%lf)\n", outputIndex, output[outputIndex], z);
}

int layerPredictDense(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(outputSize);
    layerDevPredictDense<<<gridSize, blockSize>>>(schema->predictOutput, schema->predictInput, schema->weights, inputSize, outputSize);
    return layerIfError(schema->layerIndex);
}

// ��relu�ĵ�����������������ѵ��������ȥ���޸ĵ�����
// ����ֻ��ʹ��һ�Σ���˲��������ͻ��ͬʱ��Լ�ڴ�
__global__ void layerDevTrainDense1(double* trainTemp, double* trainInput, double* predictOutput, int inputSize, int outputSize) {
    int index = blockIdx.x * outputSize + threadIdx.x;
    // relu����Ϊ0����1 ����1���� ����0��Ϊ0 �ȼ���if���
    if (predictOutput[index] > 0) {
        // trainTemp[index] = trainInput[index] / sqrt((double)(inputSize + 1));
        trainTemp[index] = trainInput[index] / (inputSize / 2);
        // trainTemp[index] = trainInput[index];
    } else {
        trainTemp[index] = 0;
    }
}

int layerTrainDense1(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(outputSize);
    layerDevTrainDense1<<<gridSize, blockSize>>>(schema->trainTemp, schema->trainInput, schema->predictOutput, inputSize, outputSize);
    return layerIfError(schema->layerIndex);
}

// ����ǰһ��ĵ���
__global__ void layerDevTrainDense2(double* trainOutput, double* weights, double* trainInput, int inputSize, int outputSize) {
    int trainOutputIndex = blockIdx.x * inputSize + threadIdx.x;
    int trainInputBeginIndex = blockIdx.x * outputSize;
    double value = 0;
    for (int i = 0; i < outputSize; i++) {
        // i��ͨ����
        int weightIndex = i * (inputSize + 1) // ��i��ͨ����Ȩ�ز���ҳ
            + threadIdx.x; // ������x�����

        // ��һ���ͨ�������Ͷ�Ӧλ��Ȩ�س˻�֮��
        // �������Ϊ��ʧ������ʾΪ����ʧ֮��
        value += trainInput[trainInputBeginIndex + i] * weights[weightIndex];
    }
    trainOutput[trainOutputIndex] = value;
}

int layerTrainDense2(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(inputSize);
    layerDevTrainDense2<<<gridSize, blockSize>>>(schema->trainOutput, schema->weights, schema->trainTemp, inputSize, outputSize);
    return layerIfError(schema->layerIndex);
}

// ����w�仯��
__global__ void layerDevTrainDense3(double* dweights, double* predictInput, double* trainInput, int batchSize, int inputSize, int outputSize) {
    int dweightsIndex = blockIdx.x * (inputSize + 1) // ����ҳ�±���ʼλ��
        + 1 + threadIdx.x; // 1��ƫ�� �ټ�w��λ��
    double value = 0;
    for (int i = 0; i < batchSize; i++) {
        // i�ǿ�id��ÿһ��w��һ��������Ȼֻ����һ��
        int trainInputIndex = i * outputSize // ��i��block��Ӧ�����ҳ
            + blockIdx.x; // ֻȡ��ǰͨ���ĵ���
        value += trainInput[trainInputIndex] * predictInput[i * inputSize + threadIdx.x]; // ����w��Ӧλ�õ�x
    }
    dweights[dweightsIndex] = value / batchSize;
}

int layerTrainDense3(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(outputSize);
    dim3 blockSize(inputSize);
    layerDevTrainDense3<<<gridSize, blockSize>>>(schema->dweights, schema->predictInput, schema->trainTemp, batchSize, inputSize, outputSize);
    return layerIfError(schema->layerIndex);
}

// ����b�仯��
__global__ void layerDevTrainDense4(double* dweights, double* trainInput, int batchSize, int inputSize, int outputSize) {
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

int layerTrainDense4(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(outputSize); // ͨ����
    dim3 blockSize(1);
    layerDevTrainDense4<<<gridSize, blockSize>>>(schema->dweights, schema->trainTemp, batchSize, inputSize, outputSize);
    return layerIfError(schema->layerIndex);
}

int layerTrainDense(layer_schema_t* schema, int batchSize) {
    int ret = 0;
    ret = ret || layerTrainDense1(schema, batchSize);
    ret = ret || layerTrainDense2(schema, batchSize);
    ret = ret || layerTrainDense3(schema, batchSize);
    ret = ret || layerTrainDense4(schema, batchSize);
    return ret;
}
