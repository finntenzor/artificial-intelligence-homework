/**
 * layer_core_scale.cu
 */

#include "layer_run.h"
#include "cuda.h"

__global__ void layerDevPredictScale1(double* input, double* output, int outputSize) {
    int inputBegin = blockIdx.x * outputSize;

    double value = 0;
    // �ҳ����ֵ
    for (int i = 0; i < outputSize; i++) {
        if (input[inputBegin + i] > value) {
            value = input[inputBegin + i];
        }
    }
    // �������ֵ
    output[blockIdx.x] = value;
}

int layerPredictScale1(layer_schema_t* schema, int batchSize) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(1);
    layerDevPredictScale1<<<gridSize, blockSize>>>(schema->predictInput, schema->predictTemp, outputSize);
    return layerIfError(schema->layerIndex);
}

__global__ void layerDevPredictScale2(double* input, double* maxz, double* output, int outputSize) {
    int index = blockIdx.x * outputSize + threadIdx.x;

    // �����������Լ����ڿ�ĵ���Сֵ
    if (maxz[blockIdx.x] == 0) {
        printf("layerDevPredictScale2 ZERO: blockIdx.x = %d, threadIdx.x = %d, index = %d\n", blockIdx.x, threadIdx.x, index);
    }
    output[index] = input[index] / maxz[blockIdx.x];
}

int layerPredictScale2(layer_schema_t* schema, int batchSize) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(outputSize);
    layerDevPredictScale2<<<gridSize, blockSize>>>(schema->predictInput, schema->predictTemp, schema->predictOutput, outputSize);
    return layerIfError(schema->layerIndex);
}

int layerPredictScale(layer_schema_t* schema, int batchSize) {
    int ret = 0;
    ret = ret || layerPredictScale1(schema, batchSize);
    ret = ret || layerPredictScale2(schema, batchSize);
    return ret;
}

__global__ void layerDevTrainScale(double* input, double* maxz, double* output, int outputSize, int layerIndex) {
    int index = blockIdx.x * outputSize + threadIdx.x;

    // �������ں�һ�㵼������ϵ��
    // input��trainInput
    if (maxz[blockIdx.x] == 0) {
        printf("layerDevTrainScale ZERO: blockIdx.x = %d, threadIdx.x = %d, index = %d\n", blockIdx.x, threadIdx.x, index);
    }
    output[index] = input[index] / maxz[blockIdx.x];

    // DEBUG
    // if (layerIndex == 6) {
    //     printf("blockIdx.x = %d, threadIdx.x = %d, index = %d, input at %p, input = %.12lf, maxz = %.12lf, output = %.12lf\n", blockIdx.x, threadIdx.x, index, input, input[index], maxz[blockIdx.x], output[index]);
    // }
}

int layerTrainScale(layer_schema_t* schema, int batchSize) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(outputSize);
    layerDevTrainScale<<<gridSize, blockSize>>>(schema->trainInput, schema->predictTemp, schema->trainOutput, outputSize, schema->layerIndex);
    return layerIfError(schema->layerIndex);
}
