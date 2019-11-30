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

// __global__ void layerDevPredictPooling1(layer_schema_t schema, int batchSize) {
//     // 参数总个数 = operationWidth * operationHeight * inputDepth
//     double* input = schema.predictInput;
//     double* args = schema.weights;

//     int inputRowBegin = (threadIdx.x / schema.operationHeight) * schema.operationHeight;
//     int inputColBegin = (threadIdx.y / schema.operationWidth) * schema.operationWidth;
//     int inputRowEnd = inputRowBegin + schema.operationHeight;
//     int inputColEnd = inputColBegin + schema.operationWidth;
//     int weightIndex = layerGetPoolingWeightIndex(&schema, blockIdx.y, threadIdx.x, threadIdx.y);

//     int maxCount = 0;
//     for (int b = 0; b < batchSize; b++) {
//         int maxRow = inputRowBegin;
//         int maxCol = inputColBegin;
//         for (int i = inputRowBegin; i < inputRowEnd; i++) { // 输入行号偏移量
//             for (int j = inputColBegin; j < inputColEnd; j++) { // 输入列号偏移量
//                 if (i < 0 || i >= schema.inputHeight || j < 0 || j >= schema.inputWidth) {
//                     continue;
//                 }
//                 int maxIndex = layerGetInputIndex(&schema, b, blockIdx.y, maxRow, maxCol);
//                 int inputIndex = layerGetInputIndex(&schema, b, blockIdx.y, i, j);
//                 if (input[inputIndex] > input[maxIndex]) {
//                     maxRow = i;
//                     maxCol = j;
//                 }
//             }
//         }
//         if (maxRow == threadIdx.x && maxCol == threadIdx.y) {
//             maxCount++;
//         }
//     }
//     args[weightIndex] = maxCount / batchSize;
// }

// int layerPredictPooling1(layer_schema_t* schema, int batchSize) {
//     dim3 gridSize(1, schema->inputDepth);
//     dim3 blockSize(schema->inputHeight, schema->inputWidth);
//     layerDevPredictPooling1<<<gridSize, blockSize>>>(*schema, batchSize);
//     return layerIfError(schema->layerIndex);
// }

// __global__ void layerDevPredictPooling2(layer_schema_t schema) {
//     // 参数总个数 = operationWidth * operationHeight * inputDepth
//     double* output = schema.predictOutput;
//     double* input = schema.predictInput;
//     double* args = schema.weights;
//     int outputIndex = layerGetCurrentOutputIndex(&schema);

//     int inputRowBegin = threadIdx.x * schema.operationHeight;
//     int inputColBegin = threadIdx.y * schema.operationWidth;
//     int inputRowEnd = inputRowBegin + schema.operationHeight;
//     int inputColEnd = inputColBegin + schema.operationWidth;

//     double z = 0;
//     for (int i = inputRowBegin; i < inputRowEnd; i++) { // 输入行号偏移量
//         for (int j = inputColBegin; j < inputColEnd; j++) { // 输入列号偏移量
//             if (i < 0 || i >= schema.inputHeight || j < 0 || j >= schema.inputWidth) {
//                 continue;
//             }
//             int inputIndex = layerGetPoolingWeightIndex(&schema, blockIdx.y, i, j);
//             int weightIndex = layerGetPoolingWeightIndex(&schema, blockIdx.y, i, j);
//             z += input[inputIndex] * args[weightIndex];
//         }
//     }
//     output[outputIndex] = z;
// }

// int layerPredictPooling2(layer_schema_t* schema, int batchSize) {
//     dim3 gridSize(batchSize, schema->outputDepth);
//     dim3 blockSize(schema->outputHeight, schema->outputWidth);
//     layerDevPredictPooling2<<<gridSize, blockSize>>>(*schema);
//     return layerIfError(schema->layerIndex);
// }

// int layerPredictPooling(layer_schema_t* schema, int batchSize) {
//     int ret = layerPredictPooling1(schema, batchSize);
//     return ret || layerPredictPooling2(schema, batchSize);
// }

__global__ void layerDevPredictPooling1(double* output, double* input,
    int windowHeight, int windowWidth,
    int rowStep, int colStep,
    int rowBasis, int colBasis,
    int inputSize, int inputChannelSize, int inputHeight, int inputWidth,
    int outputSize, int outputChannelSize, int outputHeight, int outputWidth
)
{
    double* inputBase = input + blockIdx.x * inputSize + blockIdx.y * inputChannelSize;
    double* outputBase = output + blockIdx.x * outputSize + blockIdx.y * outputChannelSize;
    int inputRowBegin = threadIdx.x * rowStep + rowBasis;
    int inputColBegin = threadIdx.y * colStep + colBasis;
    int inputRowEnd = inputRowBegin + windowHeight;
    int inputColEnd = inputColBegin + windowWidth;

    double max = inputBase[0];
    for (int i = inputRowBegin; i < inputRowEnd; i++) {
        for (int j = inputColBegin; j < inputColEnd; j++) {
            if (i < 0 || i >= inputHeight || j < 0 || inputWidth) continue;
            double curr = inputBase[i * inputWidth + j];
            if (curr > max) {
                max = curr;
            }
        }
    }
    outputBase[threadIdx.x * outputWidth + threadIdx.y] = max;
}

int layerPredictPooling1(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    int inputChannelSize = schema->inputHeight * schema->inputWidth;
    int outputChannelSize = schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize, schema->outputDepth); // 输入输出深度应该相等
    dim3 blockSize(schema->outputHeight, schema->outputWidth);
    layerDevPredictPooling1<<<gridSize, blockSize>>>(schema->predictOutput, schema->predictInput,
        schema->operationHeight, schema->operationWidth,
        schema->operationRowStep, schema->operationColStep,
        schema->operationRowBasis, schema->operationColBasis,
        inputSize, inputChannelSize, schema->inputHeight, schema->inputWidth,
        outputSize, outputChannelSize, schema->outputHeight, schema->outputWidth
    );
    return layerIfError(schema->layerIndex);
}

int layerPredictPooling(layer_schema_t* schema, int batchSize) {
    return layerPredictPooling1(schema, batchSize);
    // return ret || layerPredictPooling2(schema, batchSize);
}

__global__ void layerDevTrainPooling1(double* trainOutput, double* trainInput, double* input,
    int windowHeight, int windowWidth,
    int rowStep, int colStep,
    int rowBasis, int colBasis,
    int inputSize, int inputChannelSize, int inputHeight, int inputWidth,
    int outputSize, int outputChannelSize, int outputHeight, int outputWidth
)
{
    // double* inputBase = input + blockIdx.x * inputSize + blockIdx.y * inputChannelSize;
    // double* outputBase = output + blockIdx.x * outputSize + blockIdx.y * outputChannelSize;
    int inputRowBegin = threadIdx.x * rowStep + rowBasis - windowHeight + 1;
    int inputColBegin = threadIdx.y * colStep + colBasis - windowWidth + 1;
    int inputRowEnd = inputRowBegin + windowHeight;
    int inputColEnd = inputColBegin + windowWidth;

    // double max = inputBase[0];
    // for (int i = inputRowBegin; i < inputRowEnd; i++) {
    //     for (int j = inputColBegin; j < inputColEnd; j++) {
    //         if (i < 0 || i >= inputHeight || j < 0 || inputWidth) continue;
    //         double curr = inputBase[i * inputWidth + j];
    //         if (curr > max) {
    //             max = curr;
    //         }
    //     }
    // }
    // outputBase[threadIdx.x * outputWidth + threadIdx.y] = max;
}

int layeyTrainPooling1(layer_schema_t* schema, int batchSize) {
    int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    int inputChannelSize = schema->inputHeight * schema->inputWidth;
    int outputChannelSize = schema->outputHeight * schema->outputWidth;
    dim3 gridSize(batchSize, schema->inputDepth);
    dim3 blockSize(schema->inputHeight, schema->inputWidth);
    layerDevTrainPooling1<<<gridSize, blockSize>>>(schema->trainOutput, schema->trainInput, schema->predictInput,
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
