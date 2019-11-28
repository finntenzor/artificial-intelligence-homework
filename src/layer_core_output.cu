/**
 * layer_core_output.cu
 */

#include "layer_run.h"
#include "cuda.h"

void layerOutputGetTempPointers(layer_schema_t* schema, int batchSize, double** expx, double** expxSum, double** y) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    double* pexpx = schema->predictTemp;
    double* pexpxSum = pexpx + batchSize * outputSize;
    double* py = pexpxSum + batchSize * 1;
    // double* pnew = py + batchSize * outputSize;
    *expx = pexpx;
    *expxSum = pexpxSum;
    *y = py;
}

__global__ void layerDevPredictOutput1(double* input, double* output, int outputSize) {
    int index = blockIdx.x * outputSize + threadIdx.x;
    // 所有值转为自己关于e的指数
    output[index] = exp(input[index]);
}

int layerPredictOutput1(layer_schema_t* schema, int batchSize, double* expx) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(outputSize);
    layerDevPredictOutput1<<<gridSize, blockSize>>>(schema->predictInput, expx, outputSize);
    return layerIfError(schema->layerIndex);
}

__global__ void layerDevPredictOutput2(double* input, double* output, int outputSize) {
    // 求出每个block的和
    double sum = 0;
    for (int i = 0; i < outputSize; i++) {
        sum += input[i];
    }
    output[blockIdx.x] = sum;
}

int layerPredictOutput2(layer_schema_t* schema, int batchSize, double* expx, double* expxSum) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(1);
    layerDevPredictOutput2<<<gridSize, blockSize>>>(expx, expxSum, outputSize);
    return layerIfError(schema->layerIndex);
}

__global__ void layerDevPredictOutput3(double* expx, double* expxSum, double* y, int outputSize) {
    int index = blockIdx.x * outputSize + threadIdx.x;
    // y等于每个exp(x)除以所在block的和
    if (expxSum[blockIdx.x] == 0) {
        printf("layerDevPredictOutput3 ZERO: blockIdx.x = %d, threadIdx.x = %d, index = %d\n", blockIdx.x, threadIdx.x, index);
    }
    y[index] = expx[index] / expxSum[blockIdx.x];
}

int layerPredictOutput3(layer_schema_t* schema, int batchSize, double* expx, double* expxSum, double* y) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(outputSize);
    layerDevPredictOutput3<<<gridSize, blockSize>>>(expx, expxSum, y, outputSize);
    return layerIfError(schema->layerIndex);
}

__global__ void layerDevPredictOutput4(double* y, unsigned char* output, int outputSize) {
    // 求出每个block的最大值，返回下标
    int maxIndex = 0;
    int blockOffset = blockIdx.x * outputSize;
    for (int i = 0; i < outputSize; i++) {
        if (y[blockOffset + i] > y[blockOffset + maxIndex]) {
            maxIndex = i;
        }
    }
    output[blockIdx.x] = (unsigned char) (maxIndex);
}

int layerPredictOutput4(layer_schema_t* schema, int batchSize, double* y, unsigned char* output) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(1);
    layerDevPredictOutput4<<<gridSize, blockSize>>>(y, output, outputSize);
    return layerIfError(schema->layerIndex);
}

int layerPredictOutput(layer_schema_t* schema, int batchSize, unsigned char* output) {
    int ret = 0;
    double* expx;
    double* expxSum;
    double* y;
    layerOutputGetTempPointers(schema, batchSize, &expx, &expxSum, &y);
    ret = ret || layerPredictOutput1(schema, batchSize, expx);
    ret = ret || layerPredictOutput2(schema, batchSize, expx, expxSum);
    ret = ret || layerPredictOutput3(schema, batchSize, expx, expxSum, y);
    ret = ret || layerPredictOutput4(schema, batchSize, y, output);
    return ret;
}

__global__ void layerDevCheckOutput(double* y, unsigned char* output, unsigned char* labels, int batchSize, int outputSize, double* pacc, double* ploss) {
    int acc = 0;
    double loss = 0;
    for (int i = 0; i < batchSize; i++) {
        int label = labels[i]; // 真实值
        double kap = y[i * outputSize + label]; // 模型认为的，真实值的概率
        loss -= log(kap);
        if (output[i] == label) {
            acc++;
        }
    }
    *pacc = acc * 1.0 / batchSize;
    *ploss = loss / batchSize;
}

int layerCheckOutput(layer_schema_t* schema, int batchSize, unsigned char* output, unsigned char* labels, double* pacc, double* ploss) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    double* expx;
    double* expxSum;
    double* y;
    layerOutputGetTempPointers(schema, batchSize, &expx, &expxSum, &y);
    layerDevCheckOutput<<<1, 1>>>(y, output, labels, batchSize, outputSize, pacc, ploss);
    return layerIfError(schema->layerIndex);
}

__global__ void layerDevTrainOutput(double* y, double* output, unsigned char* labels, int outputSize) {
    int index = blockIdx.x * outputSize + threadIdx.x;
    if (labels[blockIdx.x] == threadIdx.x) { // 相当于pi
        output[index] = y[index] - 1;
    } else {
        output[index] = 0;
    }
    // printf("layerDevTrainOutput blockIdx.x = %d, threadIdx.x = %d, output at %p, index = %d, label = %d, y = %lf, output = %lf\n", blockIdx.x, threadIdx.x, output, index, labels[blockIdx.x], y[index], output[index]);
}

int layerTrainOutput(layer_schema_t* schema, int batchSize, unsigned char* labels) {
    int outputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    double* output = schema->trainOutput;
    double* expx;
    double* expxSum;
    double* y;
    layerOutputGetTempPointers(schema, batchSize, &expx, &expxSum, &y);
    dim3 gridSize(batchSize);
    dim3 blockSize(outputSize);
    layerDevTrainOutput<<<gridSize, blockSize>>>(y, output, labels, outputSize);
    return layerIfError(schema->layerIndex);
}
