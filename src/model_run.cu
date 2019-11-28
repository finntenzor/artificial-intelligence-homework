/**
 * model_run.cu
 */

#include "model_run.h"
#include "cuda.h"

static int minInt(int a, int b) {
    return a < b ? a : b;
}

int modelIfErrorWithStatus(cudaError_t cudaStatus, const char* str) {
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%s, CUDA信息: %s\n", str, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}

int modelIfError(const char* str) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "%s, CUDA信息: %s\n", str, cudaGetErrorString(cudaStatus));
        return 1;
    }
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)  {
        fprintf(stderr, "%s, CUDA信息: %s\n", str, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}

int modelCalcGridThreadCount(int total, int* block, int* thread) {
    cudaError_t cudaStatus;
    cudaDeviceProp devProp;
    int currentDevice;
    int maxThread = -1;
    int q, r;
    if (total <= 0) {
        fprintf(stderr, "总个数必须是正数，实际上是 %d\n", total);
        return 1;
    }
    cudaStatus = cudaGetDevice(&currentDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "无法获取当前运行的GPU设备号\n");
        return 1;
    }
    cudaStatus = cudaGetDeviceProperties(&devProp, currentDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "无法获取第%d个设备的设备信息\n", currentDevice);
        return 1;
    }
    maxThread = devProp.maxThreadsPerBlock;
    q = total / maxThread;
    r = total % maxThread;
    if (r > 0) q++;
    *block = q;
    *thread = maxThread;
    return 0;
}

int modelGetBatchCount(model_schema_t* mem) {
    int batchCount = mem->inputCount / mem->batchSize;
    if (mem->batchSize * batchCount < mem->inputCount) {
        batchCount++;
    }
    return batchCount;
}

int modelRunBatch(model_schema_t* mem, int offset) {
    int ret = 0;
    int inputBlockSize = modelGetInputBlockSize(mem);
    int outputBlockSize = modelGetOutputBlockSize(mem);
    int labelsBlockSize = modelGetLabelsBlockSize(mem);
    int batchSize = minInt(mem->inputCount - offset, mem->batchSize);
    unsigned char* input = mem->input + offset * inputBlockSize;
    unsigned char* output = mem->output + offset * outputBlockSize;
    unsigned char* labels = mem->labels + offset * labelsBlockSize;

    if (batchSize == 0) {
        return 0;
    }

    // printf("RUN BATCH, offset = %d\n", offset);

    for (int i = 0; i < mem->schemaCount; i++) {
        // printf("RUN LAYER %d\n", i);
        layer_schema_t* schema = &mem->schemas[i];
        switch (schema->type) {
        case LAYER_TYPE_INPUT:
            ret = layerPredictInput(schema, batchSize, input);
            break;
        case LAYER_TYPE_CONVOLUTION:
            ret = layerPredictConvolution(schema, batchSize);
            break;
        case LAYER_TYPE_POOLING:
            ret = layerPredictPooling(schema, batchSize);
            break;
        case LAYER_TYPE_DENSE:
            ret = layerPredictDense(schema, batchSize);
            break;
        case LAYER_TYPE_SCALE:
            ret = layerPredictScale(schema, batchSize);
            break;
        case LAYER_TYPE_OUTPUT:
            ret = layerPredictOutput(schema, batchSize, output);
            ret = ret || layerCheckOutput(schema, batchSize, output, labels, mem->accuracyRate, mem->loss);
            break;
        }
        if (ret) break;
    }
    return ret;
}

int modelFetchAccuracyRate(model_schema_t* mem, double* acc) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaMemcpy(acc, mem->accuracyRate, 1 * sizeof(double), cudaMemcpyDeviceToHost);
    if (modelIfErrorWithStatus(cudaStatus, "无法将准确率从显存拷贝回内存")) return 1;
    return 0;
}

int modelFetchLoss(model_schema_t* mem, double* loss) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaMemcpy(loss, mem->loss, 1 * sizeof(double), cudaMemcpyDeviceToHost);
    if (modelIfErrorWithStatus(cudaStatus, "无法将损失从显存拷贝回内存")) return 1;
    return 0;
}

// __global__ void modelDevGetBatchAccuracyCount(double* accuracyCount, unsigned char* output, unsigned char* labels, int size) {
//     int accCount = 0;
//     for (int i = 0; i < size; i++) {
//         int label = labels[i];
//         int predict = output[i];
//         if (predict == label) {
//             accCount++;
//         }
//     }
//     *accuracyCount = accCount;
// }

// int modelGetBatchAccuracy(model_schema_t* mem, int offset, double* accuracyCount, int* size) {
//     cudaError_t cudaStatus = cudaSuccess;
//     double acc = 0;
//     int outputBlockSize = modelGetOutputBlockSize(mem);
//     int labelsBlockSize = modelGetLabelsBlockSize(mem);
//     int batchSize = minInt(mem->inputCount - offset, mem->batchSize);
//     unsigned char* output = mem->output + offset * outputBlockSize;
//     unsigned char* labels = mem->labels + offset * labelsBlockSize;

//     if (batchSize == 0) {
//         return -1;
//     }

//     modelDevGetBatchAccuracyCount<<<1, 1>>>(mem->accuracyRate, output, labels, batchSize);
//     if (modelIfError("计算模型准确率时发生错误")) return 1;

//     cudaStatus = cudaMemcpy(&acc, mem->accuracyRate, 1 * sizeof(double), cudaMemcpyDeviceToHost);
//     if (modelIfErrorWithStatus(cudaStatus, "无法将准确率从显存拷贝回内存")) return 2;

//     *accuracyCount = acc;
//     *size = batchSize;
//     return 0;
// }

// double modelGetBatchAccuracyRate(model_schema_t* mem, int offset) {
//     double acc;
//     int size;
//     int ret = modelGetBatchAccuracy(mem, offset, &acc, &size);
//     return (ret) ? (-1) : (acc / size);
// }

// double modelGetAccuracyRate(model_schema_t* mem) {
//     int batchCount = modelGetBatchCount(mem);
//     double accCount = 0;
//     for (int i = 0; i < batchCount; i++) {
//         double acc;
//         int size;
//         int ret = modelGetBatchAccuracy(mem, mem->batchSize, &acc, &size);
//         if (ret) {
//             accCount = -1;
//             break;
//         } else {
//             accCount += acc;
//         }
//     }
//     return (accCount >= 0) ? (accCount / mem->inputCount) : (-1);
// }

int modelPredict(model_schema_t* mem) {
    int ret = 0;
    int batchCount = modelGetBatchCount(mem);
    for (int i = 0; i < batchCount; i++) {
        ret = ret || modelRunBatch(mem, i * mem->batchSize);
        if (ret) break;
    }
    return ret;
}

__global__ void modelDevClearDweights(double* output, int dwsize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dwsize) {
        output[index] = 0;
    }
}

int modelClearDweights(model_schema_t* mem) {
    int dwsize = modelGetDweightsSize(mem);
    int block, thread;
    modelCalcGridThreadCount(dwsize, &block, &thread);
    dim3 gridSize(block);
    dim3 blockSize(thread);
    modelDevClearDweights<<<gridSize, blockSize>>>(mem->dweights, dwsize);
    if (modelIfError("清空模型权重变化量时发生错误")) return 1;
    return 0;
}

int modelTrainBatch(model_schema_t* mem, int offset) {
    int ret = 0;
    int labelsBlockSize = modelGetLabelsBlockSize(mem);
    int batchSize = minInt(mem->inputCount - offset, mem->batchSize);
    unsigned char* labels = mem->labels + offset * labelsBlockSize;

    if (batchSize == 0) {
        return 0;
    }

    for (int i = mem->schemaCount - 1; i >= 0 ; i--) {
        layer_schema_t* schema = &mem->schemas[i];
        switch (schema->type) {
        // case LAYER_TYPE_INPUT:
        //     ret = layerPredictInput(schema, batchSize, input);
        //     break;
        // case LAYER_TYPE_CONVOLUTION:
        //     ret = layerPredictConvolution(schema, batchSize);
        //     break;
        // case LAYER_TYPE_POOLING:
        //     ret = layerPredictPooling(schema, batchSize);
        //     break;
        case LAYER_TYPE_DENSE:
            ret = layerTrainDense(schema, batchSize);
            break;
        case LAYER_TYPE_SCALE:
            ret = layerTrainScale(schema, batchSize);
            break;
        case LAYER_TYPE_OUTPUT:
            ret = layerTrainOutput(schema, batchSize, labels);
            break;
        }
        if (ret) break;
    }
    return ret;
}

__global__ void modelDevApplyDweights(double studyRate, double* weights, double* dweights, int wsize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < wsize) {
        weights[index] -= studyRate * dweights[index];
        // printf("blockIdx.x = %d, threadIdx.x = %d, index = %d, studyRate = %lf, dweights = %lf, weights = %lf\n", blockIdx.x, threadIdx.x, index, studyRate, dweights[threadIdx.x], weights[threadIdx.x]);
    }
}

int modelApplyDweights(model_schema_t* mem, int offset) {
    int wsize = modelGetWeightsSize(mem);
    int block, thread;
    int batchSize = minInt(mem->inputCount - offset, mem->batchSize);
    modelCalcGridThreadCount(wsize, &block, &thread);
    dim3 gridSize(block);
    dim3 blockSize(thread);
    modelDevApplyDweights<<<gridSize, blockSize>>>(mem->studyRate, mem->weights, mem->dweights, wsize);
    if (modelIfError("修正模型权重时发生错误")) return 1;
    return 0;
}

int modelTrain(model_schema_t* mem, int (*batchCallback)(model_schema_t* mem, int batchIndex, int step)) {
    int ret = 0;
    int batchCount = modelGetBatchCount(mem);
    ret = modelClearDweights(mem);
    double accuracyRate = 0;
    double loss = 0;

    // TODO 将ep改为loss判断
    for (int ep = 0; !ret && ep < 10; ep++) {
        for (int i = 0; !ret && i < batchCount; i++) {
            int offset = i * mem->batchSize;
            if (batchCallback != NULL) {
                ret = ret || (*batchCallback)(mem, i, 0);
            }
            ret = ret || modelRunBatch(mem, offset);
            if (batchCallback != NULL) {
                ret = ret || (*batchCallback)(mem, i, 1);
            }
            ret = ret || modelTrainBatch(mem, offset);
            if (batchCallback != NULL) {
                ret = ret || (*batchCallback)(mem, i, 2);
            }
            ret = ret || modelApplyDweights(mem, offset);
            if (batchCallback != NULL) {
                ret = ret || (*batchCallback)(mem, i, 3);
            }
            ret = ret || modelFetchAccuracyRate(mem, &accuracyRate);
            ret = ret || modelFetchLoss(mem, &loss);
            printf("当前正确率 = %10.6lf%%, 损失 = %.6lf\n", accuracyRate * 100, loss);
        }
    }
    return ret;
}
