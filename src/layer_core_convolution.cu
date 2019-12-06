/**
 * layer_core_convolution.cu
 */

#include "layer_run.h"
#include "cuda.h"

__global__ void layerDevPredictConvolution(double* output, double* input, double* weights,
    int kernelHeight, int kernelWidth, int inputChannels,
    int rowStep, int colStep,
    int rowBasis, int colBasis,
    int inputSize, int inputChannelSize, int inputHeight, int inputWidth,
    int outputSize, int outputChannelSize, int outputHeight, int outputWidth,
    int weightsSize
)
{
    double* outputBase = output + blockIdx.x * outputSize + blockIdx.y * outputChannelSize;
    double* inputBase = input + blockIdx.x * inputSize;
    double* weightsBase = weights + blockIdx.y * weightsSize;
    // double* b = weightsBase;
    double* w = weightsBase + 1;

    int inputRowBegin = threadIdx.x * rowStep + rowBasis;
    int inputColBegin = threadIdx.y * colStep + colBasis;

    double z = *weightsBase;
    for (int d = 0; d < inputChannels; d++) {
        for (int kRow = 0; kRow < kernelHeight; kRow++) {
            int xRow = inputRowBegin + kRow;
            if (xRow < 0 || xRow >= inputHeight) continue;
            for (int kCol = 0; kCol < kernelWidth; kCol++) {
                int xCol = inputColBegin + kCol;
                if (xCol < 0 || xCol >= inputWidth) continue;
                int xIndex = (d * inputHeight + xRow) * inputWidth + xCol;
                int wIndex = (d * kernelHeight + kRow) * kernelWidth + kCol;
                z += w[wIndex] * inputBase[xIndex];
            }
        }
    }

    outputBase[threadIdx.x * blockDim.y + threadIdx.y] = z;
}

int layerPredictConvolution(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    int inputChannelSize = schema->inputHeight * schema->inputWidth;
    int outputChannelSize = schema->outputHeight * schema->outputWidth;
    int weightsSize = schema->inputDepth * schema->operationHeight * schema->operationWidth + 1;
    dim3 gridSize(batchSize, schema->outputDepth);
    dim3 blockSize(schema->outputHeight, schema->outputWidth);
    layerDevPredictConvolution<<<gridSize, blockSize>>>(schema->predictOutput, schema->predictInput, schema->weights,
        schema->operationHeight, schema->operationWidth, schema->inputDepth,
        schema->operationRowStep, schema->operationColStep,
        schema->operationRowBasis, schema->operationColBasis,
        inputSize, inputChannelSize, schema->inputHeight, schema->inputWidth,
        outputSize, outputChannelSize, schema->outputHeight, schema->outputWidth,
        weightsSize
    );
    return layerIfError(schema->layerIndex);
}

__device__ int layerConvolutionBackwardIndexBegin(int index, int basis, int window, int step) {
    // 对于被除数为负数时计算结果有误，但是由于不访问对应下标的单元，结果是一致的
    return (index - basis - window + step) / step;
}

__device__ int layerConvolutionBackwardIndexEnd(int index, int basis, int window, int step) {
    return (index - basis - 0) / step;
}

// 计算前一层的导数
__global__ void layerDevTrainConvolution1(double* trainOutput, double* trainInput, double* weights,
    int kernelHeight, int kernelWidth, int outputChannels,
    int rowStep, int colStep,
    int rowBasis, int colBasis,
    int inputSize, int inputChannelSize, int inputHeight, int inputWidth,
    int outputSize, int outputChannelSize, int outputHeight, int outputWidth,
    int weightsSize, double inputRange
)
{
    double* trainInputBaseBase = trainInput + blockIdx.x * outputSize;

    int rowBegin = layerConvolutionBackwardIndexBegin(threadIdx.x, rowBasis, kernelHeight, rowStep);
    int rowEnd = layerConvolutionBackwardIndexEnd(threadIdx.x, rowBasis, kernelHeight, rowStep);
    int colBegin = layerConvolutionBackwardIndexBegin(threadIdx.y, colBasis, kernelWidth, colStep);
    int colEnd = layerConvolutionBackwardIndexEnd(threadIdx.y, colBasis, kernelWidth, colStep);
    if (rowBegin < 0) rowBegin = 0;
    if (rowEnd > inputHeight) rowBegin = inputHeight;
    if (colBegin < 0) colBegin = 0;
    if (colEnd > inputHeight) colBegin = inputHeight;

    double v = 0;
    for (int k = 0; k < outputChannels; k++) {
        double* trainInputBase = trainInputBaseBase + k * outputChannelSize;
        double* w = weights + k * weightsSize + 1 + blockIdx.y * kernelHeight * kernelWidth;
        for (int i = rowBegin; i <= rowEnd; i++) {
            for (int j = colBegin; j <= colEnd; j++) {
                int rowOffset = threadIdx.x - (i * rowStep + rowBasis);
                int colOffset = threadIdx.y - (j * colStep + colBasis);
                double dprev = trainInputBase[i * outputWidth + j];
                double dtemp = w[rowOffset * kernelWidth + colOffset];
                v += dprev * dtemp;
            }
        }
    }
    // v /= inputRange;
    trainOutput[blockIdx.x * inputSize + blockIdx.y * inputChannelSize + threadIdx.x * inputWidth + threadIdx.y] = v;
}

int layerTrainConvolution1(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    int inputChannelSize = schema->inputHeight * schema->inputWidth;
    int outputChannelSize = schema->outputHeight * schema->outputWidth;
    int weightsSize = schema->inputDepth * schema->operationHeight * schema->operationWidth + 1;
    dim3 gridSize(batchSize, schema->inputDepth);
    dim3 blockSize(schema->inputHeight, schema->inputWidth);
    layerDevTrainConvolution1<<<gridSize, blockSize>>>(schema->trainOutput, schema->trainInput, schema->weights,
        schema->operationHeight, schema->operationWidth, schema->outputDepth,
        schema->operationRowStep, schema->operationColStep,
        schema->operationRowBasis, schema->operationColBasis,
        inputSize, inputChannelSize, schema->inputHeight, schema->inputWidth,
        outputSize, outputChannelSize, schema->outputHeight, schema->outputWidth,
        weightsSize, schema->inputRange
    );
    return layerIfError(schema->layerIndex);
}

// 计算w变化量
__global__ void layerDevTrainConvolution2(double* dweights, double* trainInput, double* predictInput,
    int rowStep, int colStep,
    int rowBasis, int colBasis,
    int inputSize, int inputChannelSize, int inputHeight, int inputWidth,
    int outputSize, int outputChannelSize, int outputHeight, int outputWidth,
    int weightsSize, int batchSize, double inputRange
)
{
    double* wBase = dweights + blockIdx.x * weightsSize + 1;
    double v = 0;
    for (int mi = 0; mi < batchSize; mi++) {
        double* trainInputBase = trainInput + mi * outputSize + blockIdx.x * outputChannelSize;
        double* predictInputBase = predictInput + mi * inputSize + blockIdx.y * inputChannelSize;
        for (int i = 0; i < outputHeight; i++) {
            int xRow = i * rowStep + rowBasis + threadIdx.x;
            if (xRow < 0 || xRow >= inputHeight) continue;
            for (int j = 0; j < outputWidth; j++) {
                int xCol = j * colStep + colBasis + threadIdx.y;
                if (xCol < 0 || xCol >= inputWidth) continue;
                double dprev = trainInputBase[i * outputWidth + j];
                double dtemp = predictInputBase[xRow * inputWidth + xCol];
                v += dprev * dtemp;
            }
        }
    }
    v /= inputRange * batchSize;
    if (v > 1 || v < -1) {
        printf("Con:Warnning at (%d, %d, %d) %.16lf\n", blockIdx.x, threadIdx.x, threadIdx.y, v);
    }
    wBase[(blockIdx.y * blockDim.x + threadIdx.x) * blockDim.y + threadIdx.y] = v;
}

int layerTrainConvolution2(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    int inputChannelSize = schema->inputHeight * schema->inputWidth;
    int outputChannelSize = schema->outputHeight * schema->outputWidth;
    int weightsSize = schema->inputDepth * schema->operationHeight * schema->operationWidth + 1;
    dim3 gridSize(schema->outputDepth, schema->inputDepth);
    dim3 blockSize(schema->operationHeight, schema->operationWidth);
    layerDevTrainConvolution2<<<gridSize, blockSize>>>(schema->dweights, schema->trainInput, schema->predictInput,
        schema->operationRowStep, schema->operationColStep,
        schema->operationRowBasis, schema->operationColBasis,
        inputSize, inputChannelSize, schema->inputHeight, schema->inputWidth,
        outputSize, outputChannelSize, schema->outputHeight, schema->outputWidth,
        weightsSize, batchSize, schema->inputRange
    );
    return layerIfError(schema->layerIndex);
}

// 计算b变化量
__global__ void layerDevTrainConvolution3(double* dweights, double* trainInput,
    int outputSize, int outputChannelSize, int outputHeight, int outputWidth,
    int weightsSize, int batchSize, double inputRange
)
{
    double* bBase = dweights + blockIdx.x * weightsSize;
    double v = 0;
    for (int mi = 0; mi < batchSize; mi++) {
        double* trainInputBase = trainInput + mi * outputSize + blockIdx.x * outputChannelSize;
        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                v += trainInputBase[i * outputWidth + j];
            }
        }
    }
    v /= inputRange * batchSize;
    *bBase = v;
}

int layerTrainConvolution3(layer_schema_t* schema, int batchSize) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    int outputChannelSize = schema->outputHeight * schema->outputWidth;
    int weightsSize = schema->inputDepth * schema->operationHeight * schema->operationWidth + 1;
    dim3 gridSize(schema->outputDepth);
    dim3 blockSize(1);
    layerDevTrainConvolution3<<<gridSize, blockSize>>>(
        schema->dweights, schema->trainInput,
        outputSize, outputChannelSize, schema->outputHeight, schema->outputWidth,
        weightsSize, batchSize, schema->inputRange
    );
    return layerIfError(schema->layerIndex);
}

int layerTrainConvolution(layer_schema_t* schema, int batchSize) {
    int ret = 0;
    ret = ret || layerTrainConvolution1(schema, batchSize);
    ret = ret || layerTrainConvolution2(schema, batchSize);
    ret = ret || layerTrainConvolution3(schema, batchSize);
    return ret;
}
