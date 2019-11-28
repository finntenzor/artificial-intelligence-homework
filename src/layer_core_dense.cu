/**
 * layer_core_dense.cu
 */

#include "layer_run.h"
#include "cuda.h"

// 权重组下标，最大为输出深度-1
__device__ int layerGetDenseArgsIndex(layer_schema_t* schema, int depthIndex) {
    return depthIndex * (schema->inputDepth * schema->inputHeight * schema->inputWidth + 1);
}

__global__ void layerDevPredictDense(double* output, double* input, double* weights, int inputSize, int outputSize) {
    int outputIndex = blockIdx.x * outputSize + threadIdx.x;
    int weightBeginIndex = threadIdx.x * (inputSize + 1);

    // DEBUG
    // printf("block = %d, outputIndex = %d, argsBasis = %d, weightsSize = %d\n", blockIdx.x, outputIndex, argsBasis, weightsSize);

    // double z = 0; // 偏置b
    double z = weights[weightBeginIndex]; // 偏置b
    for (int i = 0; i < inputSize; i++) {
        int weightIndex = weightBeginIndex + 1 + i; // 权重w
        z += weights[weightIndex] * input[blockIdx.x * inputSize + i];

        // DEBUG
        // if (outputIndex == 74) {
        //     printf("%d: w = %lf, x = %lf, z = %lf\n", i, args[weightIndex], input[i], z);
        // }
    }
    // 除以神经元个数，进行归一化
    // z /= sqrt((double)(inputSize + 1));
    z /= (inputSize / 2);
    // relu激活函数
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

// 求relu的导数，并更新至本层训练输入中去（修改导数）
// 由于只会使用一次，因此不会引起冲突，同时节约内存
__global__ void layerDevTrainDense1(double* trainTemp, double* trainInput, double* predictOutput, int inputSize, int outputSize) {
    int index = blockIdx.x * outputSize + threadIdx.x;
    // relu导数为0或者1 乘以1不变 乘以0变为0 等价于if语句
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

// 计算前一层的导数
__global__ void layerDevTrainDense2(double* trainOutput, double* weights, double* trainInput, int inputSize, int outputSize) {
    int trainOutputIndex = blockIdx.x * inputSize + threadIdx.x;
    int trainInputBeginIndex = blockIdx.x * outputSize;
    double value = 0;
    for (int i = 0; i < outputSize; i++) {
        // i是通道号
        int weightIndex = i * (inputSize + 1) // 第i个通道的权重参数页
            + threadIdx.x; // 层输入x的序号

        // 上一层各通道导数和对应位置权重乘积之和
        // 求和是因为损失函数表示为各损失之和
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

// 计算w变化量
__global__ void layerDevTrainDense3(double* dweights, double* predictInput, double* trainInput, int batchSize, int inputSize, int outputSize) {
    int dweightsIndex = blockIdx.x * (inputSize + 1) // 参数页下标起始位置
        + 1 + threadIdx.x; // 1个偏置 再加w的位置
    double value = 0;
    for (int i = 0; i < batchSize; i++) {
        // i是块id，每一个w在一个块中显然只出现一次
        int trainInputIndex = i * outputSize // 第i个block对应的输出页
            + blockIdx.x; // 只取当前通道的导数
        value += trainInput[trainInputIndex] * predictInput[i * inputSize + threadIdx.x]; // 乘以w对应位置的x
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

// 计算b变化量
__global__ void layerDevTrainDense4(double* dweights, double* trainInput, int batchSize, int inputSize, int outputSize) {
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

int layerTrainDense4(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    dim3 gridSize(outputSize); // 通道数
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
