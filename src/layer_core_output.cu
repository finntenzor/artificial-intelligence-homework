/**
 * layer_core_output.cu
 */

#include "layer_run.h"
#include "cuda.h"

void layerOutputGetTempPointers(layer_schema_t* schema, int batchSize, double** pmaxx, double** psumex, double** pexpx, double** py) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    double* maxx = schema->predictTemp;
    double* sumex = maxx + batchSize * 1;
    double* expx = sumex + batchSize * 1;
    double* y = expx + batchSize * outputSize;
    *pmaxx = maxx;
    *psumex = sumex;
    *pexpx = expx;
    *py = y;
}

__global__ void layerDevPredictOutput1(double* x, double* maxx, int outputSize) {
    // 求出x的最大值
    int begin = blockIdx.x * outputSize;
    double max = x[begin];
    for (int i = 0; i < outputSize; i++) {
        int curr = x[begin + i];
        if (curr > max) {
            max = curr;
        }
    }
    maxx[blockIdx.x] = max;
}

int layerPredictOutput1(layer_schema_t* schema, int batchSize, double* maxx) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(1);
    layerDevPredictOutput1<<<gridSize, blockSize>>>(schema->predictInput, maxx, outputSize);
    return layerIfError(schema->layerIndex);
}

__global__ void layerDevPredictOutput2(double* x, double* maxx, double* expx, int outputSize) {
    int index = blockIdx.x * outputSize + threadIdx.x;
    // 求exp(x-a)
    expx[index] = exp(x[index] - maxx[blockIdx.x]);
}

int layerPredictOutput2(layer_schema_t* schema, int batchSize, double* maxx, double* expx) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(outputSize);
    layerDevPredictOutput2<<<gridSize, blockSize>>>(schema->predictInput, maxx, expx, outputSize);
    return layerIfError(schema->layerIndex);
}

__global__ void layerDevPredictOutput3(double* expx, double* sumex, int outputSize) {
    // 求sum(exp(x-a))
    int begin = blockIdx.x * outputSize;
    double sum = 0;
    for (int i = 0; i < outputSize; i++) {
        sum += expx[begin + i];
    }
    sumex[blockIdx.x] = sum;
}

int layerPredictOutput3(layer_schema_t* schema, int batchSize, double* expx, double* sumex) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(1);
    layerDevPredictOutput3<<<gridSize, blockSize>>>(expx, sumex, outputSize);
    return layerIfError(schema->layerIndex);
}

__global__ void layerDevPredictOutput4(double* expx, double* sumex, double* y, int outputSize) {
    int index = blockIdx.x * outputSize + threadIdx.x;
    // y等于每个exp(x-a)除以sum(exp(x-a))
    y[index] = expx[index] / sumex[blockIdx.x];
}

int layerPredictOutput4(layer_schema_t* schema, int batchSize, double* expx, double* sumex, double* y) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(outputSize);
    layerDevPredictOutput4<<<gridSize, blockSize>>>(expx, sumex, y, outputSize);
    return layerIfError(schema->layerIndex);
}

__global__ void layerDevPredictOutput5(double* x, unsigned char* output, int outputSize) {
    // 求出每个block的最大值，返回下标
    int maxIndex = 0;
    int blockOffset = blockIdx.x * outputSize;
    for (int i = 0; i < outputSize; i++) {
        if (x[blockOffset + i] > x[blockOffset + maxIndex]) {
            maxIndex = i;
        }
    }
    output[blockIdx.x] = (unsigned char) (maxIndex);
}

int layerPredictOutput5(layer_schema_t* schema, int batchSize, unsigned char* output) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize);
    dim3 blockSize(1);
    layerDevPredictOutput5<<<gridSize, blockSize>>>(schema->predictInput, output, outputSize);
    return layerIfError(schema->layerIndex);
}

int layerPredictOutput(layer_schema_t* schema, int batchSize, unsigned char* output) {
    int ret = 0;
    double* maxx;
    double* sumex;
    double* expx;
    double* y;
    layerOutputGetTempPointers(schema, batchSize, &maxx, &sumex, &expx, &y);
    ret = ret || layerPredictOutput1(schema, batchSize, maxx);
    ret = ret || layerPredictOutput2(schema, batchSize, maxx, expx);
    ret = ret || layerPredictOutput3(schema, batchSize, expx, sumex);
    ret = ret || layerPredictOutput4(schema, batchSize, expx, sumex, y);
    ret = ret || layerPredictOutput5(schema, batchSize, output);
    return ret;
}

__global__ void layerDevCheckOutput(double* y, unsigned char* output, unsigned char* labels, int batchSize, int outputSize, int* pacc, int* ptot, double* ploss) {
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
    *pacc += acc;
    *ploss += loss;
    *ptot += batchSize;
}

int layerCheckOutput(layer_schema_t* schema, int batchSize, unsigned char* output, unsigned char* labels, int* pacc, int* ptot, double* ploss) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    double* maxx;
    double* sumex;
    double* expx;
    double* y;
    layerOutputGetTempPointers(schema, batchSize, &maxx, &sumex, &expx, &y);
    layerDevCheckOutput<<<1, 1>>>(y, output, labels, batchSize, outputSize, pacc, ptot, ploss);
    return layerIfError(schema->layerIndex);
}

__global__ void layerDevTrainOutput(double* y, double* output, unsigned char* labels, int outputSize) {
    int index = blockIdx.x * outputSize + threadIdx.x;
    double y_ = (labels[blockIdx.x] == threadIdx.x) ? 1 : 0;
    output[index] = y[index] - y_;
}

int layerTrainOutput(layer_schema_t* schema, int batchSize, unsigned char* labels) {
    int outputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    double* output = schema->trainOutput;
    double* maxx;
    double* sumex;
    double* expx;
    double* y;
    layerOutputGetTempPointers(schema, batchSize, &maxx, &sumex, &expx, &y);
    dim3 gridSize(batchSize);
    dim3 blockSize(outputSize);
    layerDevTrainOutput<<<gridSize, blockSize>>>(y, output, labels, outputSize);
    return layerIfError(schema->layerIndex);
}
