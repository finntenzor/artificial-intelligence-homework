/**
 * layer_core_convolution.cu
 */

#include "layer_run.h"
#include "cuda.h"

// // 卷积核下标，最大为输出深度-1
// __device__ int layerGetConvolutionArgsIndex(layer_schema_t* schema, int kernelIndex) {
//     return kernelIndex * (schema->inputDepth * schema->operationHeight * schema->operationWidth + 1);
// }

// // 共有卷积核组权重和偏置
// __device__ int layerGetConvolutionWeightIndexOffset(layer_schema_t* schema, int channelIndex, int rowIndex, int colIndex) {
//     // 多加1是因为这个位置是偏置
//     return 1 + (channelIndex * schema->operationHeight + rowIndex) * schema->operationWidth + colIndex;
// }

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
        for (int i = 0; i < kernelHeight; i++) {
            int xRow = inputRowBegin + i;
            if (xRow <= 0 || xRow >= inputHeight) continue;
            for (int j = 0; j < kernelWidth; j++) {
                int xCol = inputColBegin + j;
                if (j <= 0 || j >= inputWidth) continue;
                int xIndex = (d * inputChannels + xRow) * inputWidth + xCol;
                int wIndex = (d * inputChannels + i) * kernelWidth + j;
                z += w[wIndex] * inputBase[xIndex];
            }
        }
    }

    // relu激活函数
    outputBase[threadIdx.x * blockDim.y + threadIdx.y] = (z > 0 ? z : 0);
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

int layerTrainConvolution(layer_schema_t* schema, int batchSize) {
    // int inputSize = schema->inputDepth * schema->inputHeight * schema->inputWidth;
    // int outputSize = schema->outputDepth * schema->outputHeight * schema->outputWidth;
    // int inputChannelSize = schema->inputHeight * schema->inputWidth;
    // int outputChannelSize = schema->outputHeight * schema->outputWidth;
    // int weightsSize = schema->inputDepth * schema->operationHeight * schema->operationWidth + 1;
    // dim3 gridSize(batchSize, schema->outputDepth);
    // dim3 blockSize(schema->outputHeight, schema->outputWidth);
    // layerDevPredictConvolution<<<gridSize, blockSize>>>(schema->predictOutput, schema->predictInput, schema->weights,
    //     schema->operationHeight, schema->operationWidth, schema->inputDepth,
    //     schema->operationRowStep, schema->operationColStep,
    //     schema->operationRowBasis, schema->operationColBasis,
    //     inputSize, inputChannelSize, schema->inputHeight, schema->inputWidth,
    //     outputSize, outputChannelSize, schema->outputHeight, schema->outputWidth,
    //     weightsSize
    // );
    return layerIfError(schema->layerIndex);
}
