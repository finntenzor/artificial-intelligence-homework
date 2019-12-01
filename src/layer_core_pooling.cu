/**
 * layer_core_pooling.cu
 */

#include "layer_run.h"
#include "cuda.h"

// // 仅一组权重
// __device__ int layerGetPoolingWeightIndex(layer_schema_t* schema, int channelIndex, int rowIndex, int colIndex) {
//     // 不加1，没有偏置
//     return (channelIndex * schema->inputHeight + rowIndex) * schema->inputWidth + colIndex;
// }

__device__ int layerPoolingBackwardIndexBegin(int index, int basis, int window, int step) {
    // 对于被除数为负数时计算结果有误，但是由于不访问对应下标的单元，结果是一致的
    return (index - basis - window + step) / step;
}

__device__ int layerPoolingBackwardIndexEnd(int index, int basis, int window, int step) {
    return (index - basis - 0) / step;
}

__global__ void layerDevPredictPooling1(double* output, double* input, double* temp,
    int windowHeight, int windowWidth,
    int rowStep, int colStep,
    int rowBasis, int colBasis,
    int inputSize, int inputChannelSize, int inputHeight, int inputWidth,
    int outputSize, int outputChannelSize, int outputHeight, int outputWidth,
    int tempSize, int tempChannelSize
)
{
    double* inputBase = input + blockIdx.x * inputSize + blockIdx.y * inputChannelSize;
    double* outputBase = output + blockIdx.x * outputSize + blockIdx.y * outputChannelSize;
    double* tempBase = temp + blockIdx.x * tempSize + blockIdx.y * tempChannelSize;
    int inputRowBegin = threadIdx.x * rowStep + rowBasis;
    int inputColBegin = threadIdx.y * colStep + colBasis;
    int inputRowEnd = inputRowBegin + windowHeight;
    int inputColEnd = inputColBegin + windowWidth;
    int rowBase = threadIdx.x * windowHeight;
    int colBase = threadIdx.y * windowWidth;
    int rowOffset = inputRowBegin;
    int colOffset = inputColBegin;

    double max = inputBase[0];
    for (int i = inputRowBegin; i < inputRowEnd; i++) {
        if (i < 0 || i >= inputHeight) continue;
        for (int j = inputColBegin; j < inputColEnd; j++) {
            if (j < 0 || j >= inputWidth) continue;
            double curr = inputBase[i * inputWidth + j];
            if (curr > max) {
                max = curr;
                rowOffset = i;
                colOffset = j;
            }
        }
    }
    for (int i = 0; i < windowHeight; i++) {
        for (int j = 0; j < windowWidth; j++) {
            int index = (rowBase + i) * (outputWidth * windowWidth) + colBase + j;
            temp[index] = 0;
        }
    }
    rowOffset -= inputRowBegin;
    colOffset -= inputColBegin;
    outputBase[threadIdx.x * outputWidth + threadIdx.y] = max;
    tempBase[(rowBase + rowOffset) * (outputWidth * windowWidth) + colBase + colOffset] = 1.0 / (windowWidth * windowHeight);
}

int layerPredictPooling1(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    int tempSize = schema->outputDepth * schema->outputHeight * schema->outputWidth * schema->operationHeight * schema->operationWidth;
    int inputChannelSize = schema->inputHeight * schema->inputWidth;
    int outputChannelSize = schema->outputHeight * schema->outputWidth;
    int tempChannelSize = schema->outputHeight * schema->outputWidth * schema->operationHeight * schema->operationWidth;
    dim3 gridSize(batchSize, schema->outputDepth); // 输入输出深度应该相等
    dim3 blockSize(schema->outputHeight, schema->outputWidth);
    layerDevPredictPooling1<<<gridSize, blockSize>>>(schema->predictOutput, schema->predictInput, schema->predictTemp,
        schema->operationHeight, schema->operationWidth,
        schema->operationRowStep, schema->operationColStep,
        schema->operationRowBasis, schema->operationColBasis,
        inputSize, inputChannelSize, schema->inputHeight, schema->inputWidth,
        outputSize, outputChannelSize, schema->outputHeight, schema->outputWidth,
        tempSize, tempChannelSize
    );
    return layerIfError(schema->layerIndex);
}

int layerPredictPooling(layer_schema_t* schema, int batchSize) {
    return layerPredictPooling1(schema, batchSize);
}

__global__ void layerDevTrainPooling1(double* trainOutput, double* trainInput, double* predictTemp,
    int windowHeight, int windowWidth,
    int rowStep, int colStep,
    int rowBasis, int colBasis,
    int inputSize, int inputChannelSize, int inputHeight, int inputWidth,
    int outputSize, int outputChannelSize, int outputHeight, int outputWidth
)
{
    double* trainOutputBase = trainOutput + blockIdx.x * inputSize + blockIdx.y * inputChannelSize;
    double* trainInputBase = trainInput + blockIdx.x * outputSize + blockIdx.y * outputChannelSize;

    int rowBegin = layerPoolingBackwardIndexBegin(threadIdx.x, rowBasis, windowHeight, rowStep);
    int rowEnd = layerPoolingBackwardIndexEnd(threadIdx.x, rowBasis, windowHeight, rowStep);
    int colBegin = layerPoolingBackwardIndexBegin(threadIdx.y, colBasis, windowWidth, colStep);
    int colEnd = layerPoolingBackwardIndexEnd(threadIdx.y, colBasis, windowWidth, colStep);

    double v = 0;
    predictTemp += (blockIdx.x * outputSize + blockIdx.y) * windowHeight * windowWidth;
    for (int i = rowBegin; i <= rowEnd; i++) {
        if (i < 0 || i >= outputHeight) continue;
        for (int j = colBegin; j <= colEnd; j++) {
            if (j < 0 || j >= outputWidth) continue;
            int rowOffset = threadIdx.x - (i * rowStep + rowBasis);
            int colOffset = threadIdx.y - (j * colStep + colBasis);
            double dprev = trainInputBase[i * outputWidth + j];
            double dtemp = predictTemp[(i * windowHeight + rowOffset) * (outputWidth * windowWidth) + j * windowWidth + colOffset];
            v += dprev * dtemp;
        }
    }
    trainOutputBase[threadIdx.x * inputWidth + threadIdx.y] = v;
}

int layeyTrainPooling1(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    int inputChannelSize = schema->inputHeight * schema->inputWidth;
    int outputChannelSize = schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize, schema->inputDepth);
    dim3 blockSize(schema->inputHeight, schema->inputWidth);
    layerDevTrainPooling1<<<gridSize, blockSize>>>(schema->trainOutput, schema->trainInput, schema->predictTemp,
        schema->operationHeight, schema->operationWidth,
        schema->operationRowStep, schema->operationColStep,
        schema->operationRowBasis, schema->operationColBasis,
        inputSize, inputChannelSize, schema->inputHeight, schema->inputWidth,
        outputSize, outputChannelSize, schema->outputHeight, schema->outputWidth
    );
    return layerIfError(schema->layerIndex);
}

int layerTrainPooling(layer_schema_t* schema, int batchSize) {
    return layeyTrainPooling1(schema, batchSize);
}
