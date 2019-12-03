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

    // relu
    outputBase[threadIdx.x * blockDim.y + threadIdx.y] = (z > 0 ? z : z * 0.01);
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

// 求relu的导数，保存至临时变量
__global__ void layerDevTrainConvolution1(double* trainInput, double* trainTemp, double* predictOutput) {
    int index = ((blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x) * blockDim.y + threadIdx.y;
    // relu导数为0或者1 乘以1不变 乘以0变为0 等价于if语句
    if (predictOutput[index] > 0) {
        trainTemp[index] = trainInput[index];
    } else {
        trainTemp[index] = trainInput[index] * 0.01;
    }
}

int layerTrainConvolution1(layer_schema_t* schema, int batchSize) {
    // int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize, schema->outputDepth);
    dim3 blockSize(schema->outputHeight, schema->outputWidth);
    layerDevTrainConvolution1<<<gridSize, blockSize>>>(schema->trainInput, schema->trainTemp, schema->predictOutput);
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
__global__ void layerDevTrainConvolution2(double* trainOutput, double* trainTemp, double* weights,
    int kernelHeight, int kernelWidth, int outputChannels,
    int rowStep, int colStep,
    int rowBasis, int colBasis,
    int inputSize, int inputChannelSize, int inputHeight, int inputWidth,
    int outputSize, int outputChannelSize, int outputHeight, int outputWidth,
    int weightsSize
)
{
    double* trainTempBaseBase = trainTemp + blockIdx.x * outputSize;

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
        double* trainTempBase = trainTempBaseBase + k * outputChannelSize;
        double* w = weights + k * weightsSize + 1;
        for (int i = rowBegin; i < rowEnd; i++) {
            for (int j = colBegin; j < colEnd; j++) {
                int rowOffset = threadIdx.x - (i * rowStep + rowBasis);
                int colOffset = threadIdx.y - (j * colStep + colBasis);
                double dprev = trainTempBase[i * outputWidth + j];
                double dtemp = w[rowOffset * kernelWidth + colOffset];
                v += dprev * dtemp;
            }
        }
    }
    trainOutput[blockIdx.x * inputSize + blockIdx.y * inputChannelSize + threadIdx.x * inputWidth + threadIdx.y] = v;
}

int layerTrainConvolution2(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    int inputChannelSize = schema->inputHeight * schema->inputWidth;
    int outputChannelSize = schema->outputHeight * schema->outputWidth;
    int weightsSize = schema->inputDepth * schema->operationHeight * schema->operationWidth + 1;
    dim3 gridSize(batchSize, schema->inputDepth);
    dim3 blockSize(schema->inputHeight, schema->inputWidth);
    layerDevTrainConvolution2<<<gridSize, blockSize>>>(schema->trainOutput, schema->trainTemp, schema->weights,
        schema->operationHeight, schema->operationWidth, schema->outputDepth,
        schema->operationRowStep, schema->operationColStep,
        schema->operationRowBasis, schema->operationColBasis,
        inputSize, inputChannelSize, schema->inputHeight, schema->inputWidth,
        outputSize, outputChannelSize, schema->outputHeight, schema->outputWidth,
        weightsSize
    );
    return layerIfError(schema->layerIndex);
}

// 计算w变化量
__global__ void layerDevTrainConvolution3(double* dweights, double* trainTemp, double* predictInput,
    int rowStep, int colStep,
    int rowBasis, int colBasis,
    int inputSize, int inputChannelSize, int inputHeight, int inputWidth,
    int outputSize, int outputChannelSize, int outputHeight, int outputWidth,
    int weightsSize, int batchSize
)
{
    double* wBase = dweights + blockIdx.x * weightsSize + 1;
    double v = 0;
    for (int mi = 0; mi < batchSize; mi++) {
        double* trainTempBase = trainTemp + mi * outputSize + blockIdx.x * outputChannelSize;
        double* predictInputBase = predictInput + mi * inputSize + blockIdx.x * inputChannelSize;
        for (int i = 0; i < outputHeight; i++) {
            int xRow = i * rowStep + rowBasis + threadIdx.x;
            if (xRow < 0 || xRow >= inputHeight) continue;
            for (int j = 0; j < outputWidth; j++) {
                int xCol = j * colStep + colBasis + threadIdx.y;
                if (xCol < 0 || xCol >= inputWidth) continue;
                double dprev = trainTempBase[i * outputWidth + j];
                double dtemp = predictInputBase[xRow * inputWidth + xCol];
                v += dprev * dtemp;
            }
        }
    }
    wBase[threadIdx.x * blockDim.x + threadIdx.y] = v / batchSize;
}

int layerTrainConvolution3(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    int inputChannelSize = schema->inputHeight * schema->inputWidth;
    int outputChannelSize = schema->outputHeight * schema->outputWidth;
    int weightsSize = schema->inputDepth * schema->operationHeight * schema->operationWidth + 1;
    dim3 gridSize(schema->outputDepth);
    dim3 blockSize(schema->operationHeight, schema->operationWidth);
    layerDevTrainConvolution3<<<gridSize, blockSize>>>(schema->dweights, schema->trainTemp, schema->predictInput,
        schema->operationRowStep, schema->operationColStep,
        schema->operationRowBasis, schema->operationColBasis,
        inputSize, inputChannelSize, schema->inputHeight, schema->inputWidth,
        outputSize, outputChannelSize, schema->outputHeight, schema->outputWidth,
        weightsSize, batchSize
    );
    return layerIfError(schema->layerIndex);
}

// 计算b变化量
__global__ void layerDevTrainConvolution4(double* dweights, double* trainTemp,
    int outputSize, int outputChannelSize, int weightsSize, int batchSize
)
{
    double* bBase = dweights + blockIdx.x * weightsSize;
    double v = 0;
    for (int mi = 0; mi < batchSize; mi++) {
        v += trainTemp[mi * outputSize + blockIdx.x * outputChannelSize + threadIdx.x];
    }
    *bBase = v / batchSize;
}

int layerTrainConvolution4(layer_schema_t* schema, int batchSize) {
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    int outputChannelSize = schema->outputHeight * schema->outputWidth;
    int weightsSize = schema->inputDepth * schema->operationHeight * schema->operationWidth + 1;
    dim3 gridSize(schema->outputDepth);
    dim3 blockSize(schema->outputHeight * schema->outputWidth);
    layerDevTrainConvolution4<<<gridSize, blockSize>>>(
        schema->dweights, schema->trainTemp,
        outputSize, outputChannelSize, weightsSize, batchSize
    );
    return layerIfError(schema->layerIndex);
}

int layerTrainConvolution(layer_schema_t* schema, int batchSize) {
    int ret = 0;
    ret = ret || layerTrainConvolution1(schema, batchSize);
    ret = ret || layerTrainConvolution2(schema, batchSize);
    ret = ret || layerTrainConvolution3(schema, batchSize);
    ret = ret || layerTrainConvolution4(schema, batchSize);
    return ret;
}
