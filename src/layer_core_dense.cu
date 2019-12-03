/**
 * layer_core_dense.cu
 */

#include "layer_run.h"
#include "cuda.h"

// 权重组下标，最大为输出深度-1
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

    double z = weights[weightBeginIndex]; // 偏置b
    for (int i = 0; i < inputSize; i++) {
        int weightIndex = weightBeginIndex + 1 + i; // 权重w
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

// 计算前一层的导数
__global__ void layerDevTrainDense1(double* trainOutput, double* weights, double* trainInput, int outputSize, int weightsSize) {
    int xIndex = blockIdx.y * blockDim.x + threadIdx.x;
    int trainOutputIndex = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
    int trainInputBeginIndex = blockIdx.x * outputSize;
    double value = 0;
    for (int i = 0; i < outputSize; i++) {
        // i是通道号
        int weightIndex = i * weightsSize// 第i个通道的权重参数页
            + 1 + xIndex; // 层输入x的序号

        // 上一层各通道导数和对应位置权重乘积之和
        // 求和是因为损失函数表示为各损失之和
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

// 计算w变化量
__global__ void layerDevTrainDense2(double* dweights, double* predictInput, double* trainInput, int batchSize, int inputSize, int outputSize) {
    int xIndex = blockIdx.y * blockDim.x + threadIdx.x;
    int dweightsIndex = blockIdx.x * (inputSize + 1) // 参数页下标起始位置
        + 1 + xIndex; // 1个偏置 再加w的位置
    double value = 0;
    for (int i = 0; i < batchSize; i++) {
        // i是块id，每一个w在一个块中显然只出现一次
        int trainInputIndex = i * outputSize // 第i个block对应的输出页
            + blockIdx.x; // 只取当前通道的导数
        value += trainInput[trainInputIndex] * predictInput[i * inputSize + xIndex]; // 乘以w对应位置的x
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

// 计算b变化量
__global__ void layerDevTrainDense3(double* dweights, double* trainInput, int batchSize, int inputSize, int outputSize) {
    int dweightsIndex = blockIdx.x * (inputSize + 1); // 参数页下标起始位置 下标以偏置开始，加0
    double value = 0;
    for (int i = 0; i < batchSize; i++) {
        // i是块id，每一个b在一个块中显然只出现一次
        int trainInputIndex = i * outputSize // 第i个block对应的输出页
            + blockIdx.x; // 只取当前通道的导数
        value += trainInput[trainInputIndex]; // 由于b是常数 对b的导数等于1 直接把上一层导数求和即可
    }
    dweights[dweightsIndex] = value / batchSize;
}

int layerTrainDense3(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(outputSize); // 通道数
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
