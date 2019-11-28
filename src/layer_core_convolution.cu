/**
 * layer_core_convolution.cu
 */

#include "layer_run.h"
#include "cuda.h"

// ������±꣬���Ϊ������-1
__device__ int layerGetConvolutionArgsIndex(layer_schema_t* schema, int kernelIndex) {
    return kernelIndex * (schema->inputDepth * schema->operationHeight * schema->operationWidth + 1);
}

// ���о������Ȩ�غ�ƫ��
__device__ int layerGetConvolutionWeightIndexOffset(layer_schema_t* schema, int channelIndex, int rowIndex, int colIndex) {
    // ���1����Ϊ���λ����ƫ��
    return 1 + (channelIndex * schema->operationHeight + rowIndex) * schema->operationWidth + colIndex;
}

__global__ void layerDevPredictConvolution(layer_schema_t schema) {
    double* output = schema.predictOutput;
    double* input = schema.predictInput;
    double* args = schema.weights;
    // �����ܸ��� = outputDepth * (operationWidth * operationHeight * inputDepth + 1)
    int outputIndex = layerGetCurrentOutputIndex(&schema);
    int argsBasis = layerGetConvolutionArgsIndex(&schema, blockIdx.y);

    int kernelWidth = (int)schema.operationWidth;
    int kernelHeight = (int)schema.operationHeight;
    int rowBegin = threadIdx.x * schema.operationRowStep + schema.operationRowBasis;
    int colBegin = threadIdx.y * schema.operationColStep + schema.operationColBasis;

    double z = args[argsBasis]; // ƫ��
    for (int k = 0; k < schema.inputDepth; k++) { // ����ͨ����
        for (int i = 0; i < kernelHeight; i++) { // �����к�ƫ����
            for (int j = 0; j < kernelWidth; j++) { // �����к�ƫ����
                int row = rowBegin + i; // �����к�
                int col = colBegin + j; // �����к�
                int inputIndex = layerGetInputIndex(&schema, blockIdx.x, k, row, col);
                int weightIndex = argsBasis + layerGetConvolutionWeightIndexOffset(&schema, k, i, j);
                if (row < 0 || row >= schema.inputHeight || col < 0 || col >= schema.inputWidth) {
                    continue;
                }
                z += args[weightIndex] * input[inputIndex];
            }
        }
    }
    // relu�����
    output[outputIndex] = (z > 0 ? z : 0);
}

int layerPredictConvolution(layer_schema_t* schema, int batchSize) {
    dim3 gridSize(batchSize, schema->outputDepth);
    dim3 blockSize(schema->outputHeight, schema->outputWidth);
    layerDevPredictConvolution<<<gridSize, blockSize>>>(*schema);
    return layerIfError(schema->layerIndex);
}
