/**
 * model_run.cu
 */

#include "model_run.h"
#include "cuda.h"

static int minInt(int a, int b) {
    return a < b ? a : b;
}

static int ceilDivide(int a, int b) {
    int c = a / b;
    if (a > b * c) c++;;
    return c;
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

int modelRunBatch(model_schema_t* mem, int offset, int batchSize) {
    int ret = 0;
    int inputBlockSize = modelGetInputBlockSize(mem);
    int outputBlockSize = modelGetOutputBlockSize(mem);
    int labelsBlockSize = modelGetLabelsBlockSize(mem);
    unsigned char* input = mem->input + offset * inputBlockSize;
    unsigned char* output = mem->output + offset * outputBlockSize;
    unsigned char* labels = mem->labels + offset * labelsBlockSize;

    if (batchSize == 0) {
        return 0;
    }

    for (int i = 0; i < mem->schemaCount; i++) {
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
        case LAYER_TYPE_RELU:
            ret = layerPredictRelu(schema, batchSize);
            break;
        case LAYER_TYPE_TANH:
            ret = layerPredictTanh(schema, batchSize);
            break;
        case LAYER_TYPE_OUTPUT:
            ret = layerPredictOutput(schema, batchSize, output);
            ret = ret || layerCheckOutput(schema, batchSize, output, labels, mem->accuracyCount, mem->totalCount, mem->loss);
            break;
        }
        if (ret) break;
    }
    return ret;
}

int modelFetchAccuracyRate(model_schema_t* mem, int* acc) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaMemcpy(acc, mem->accuracyCount, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    if (modelIfErrorWithStatus(cudaStatus, "无法将准确个数从显存拷贝回内存")) return 1;
    return 0;
}

int modelFetchTotalCount(model_schema_t* mem, int* totalCount) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaMemcpy(totalCount, mem->totalCount, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    if (modelIfErrorWithStatus(cudaStatus, "无法将训练个数从显存拷贝回内存")) return 1;
    return 0;
}

int modelFetchLoss(model_schema_t* mem, double* loss) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaMemcpy(loss, mem->loss, 1 * sizeof(double), cudaMemcpyDeviceToHost);
    if (modelIfErrorWithStatus(cudaStatus, "无法将损失从显存拷贝回内存")) return 1;
    return 0;
}

int modelPredict(model_schema_t* mem) {
    int ret = 0;
    int batchCount = ceilDivide(mem->predictCount, mem->batchSize);
    for (int i = 0; i < batchCount; i++) {
        int offset = i * mem->batchSize;
        int batchSize = minInt(mem->predictCount - offset, mem->batchSize);
        ret = ret || modelRunBatch(mem, offset, batchSize);
        if (ret) break;
    }
    return ret;
}

__global__ void modelDevClearDweights(double* dweights, double* mweights, double* vweights, int dwsize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < dwsize) {
        dweights[index] = 0;
        mweights[index] = 0;
        vweights[index] = 0;
    }
}

__global__ void modelDevClearAccuracy(int* accuracyCount, int* totalCount, double* loss) {
    *accuracyCount = 0;
    *totalCount = 0;
    *loss = 0;
}

int modelClearDweights(model_schema_t* mem) {
    int dwsize = modelGetDweightsSize(mem);
    int block, thread;
    modelCalcGridThreadCount(dwsize, &block, &thread);
    dim3 gridSize(block);
    dim3 blockSize(thread);
    modelDevClearDweights<<<gridSize, blockSize>>>(mem->dweights, mem->mweights, mem->vweights, dwsize);
    if (modelIfError("清空模型权重变化量时发生错误")) return 1;
    return 0;
}

int modelClearAccuracy(model_schema_t* mem) {
    modelDevClearAccuracy<<<1, 1>>>(mem->accuracyCount, mem->totalCount, mem->loss);
    if (modelIfError("清空模型准确率参数时发生错误")) return 1;
    return 0;
}

int modelClearArguments(model_schema_t* mem) {
    return modelClearDweights(mem) || modelClearAccuracy(mem);
}

int modelTrainBatch(model_schema_t* mem, int offset, int batchSize) {
    int ret = 0;
    int labelsBlockSize = modelGetLabelsBlockSize(mem);
    unsigned char* labels = mem->labels + offset * labelsBlockSize;

    if (batchSize == 0) {
        return 0;
    }

    for (int i = mem->schemaCount - 1; i >= 0 ; i--) {
        layer_schema_t* schema = &mem->schemas[i];
        switch (schema->type) {
        case LAYER_TYPE_CONVOLUTION:
            ret = layerTrainConvolution(schema, batchSize);
            break;
        case LAYER_TYPE_POOLING:
            ret = layerTrainPooling(schema, batchSize);
            break;
        case LAYER_TYPE_DENSE:
            ret = layerTrainDense(schema, batchSize);
            break;
        case LAYER_TYPE_SCALE:
            ret = layerTrainScale(schema, batchSize);
            break;
        case LAYER_TYPE_RELU:
            ret = layerTrainRelu(schema, batchSize);
            break;
        case LAYER_TYPE_TANH:
            ret = layerTrainTanh(schema, batchSize);
            break;
        case LAYER_TYPE_OUTPUT:
            ret = layerTrainOutput(schema, batchSize, labels);
            break;
        }
        if (ret) break;
    }
    return ret;
}

__global__ void modelDevApplyDweights(double studyRate, double* weights, double* dweights, double* mweights, double* vweights, double t, int wsize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < wsize) {
        // Adam
        double g = dweights[index];
        double m = 0.9 * mweights[index] + 0.1 * g;
        double v = 0.999 * vweights[index] + 0.001 * g * g;
        // double w = weights[index];
        mweights[index] = m;
        vweights[index] = v;
        m /= (1 - pow(0.9, t));
        v /= (1 - pow(0.999, t));
        weights[index] -= studyRate * m / (sqrt(v) + 1e-8);

        // 梯度下降
        // weights[index] -= studyRate * dweights[index];
    }
}

int modelApplyDweights(model_schema_t* mem, int offset, double t) {
    int wsize = modelGetWeightsSize(mem);
    int block, thread;
    modelCalcGridThreadCount(wsize, &block, &thread);
    dim3 gridSize(block);
    dim3 blockSize(thread);
    modelDevApplyDweights<<<gridSize, blockSize>>>(mem->studyRate, mem->weights, mem->dweights, mem->mweights, mem->vweights, t, wsize);
    if (modelIfError("修正模型权重时发生错误")) return 1;
    return 0;
}

static double* initTestLosses(int n) {
    double* losses = new double[n];
    for (int i = 0; i < n; i++) {
        losses[i] = i + 1e10;
    }
    return losses;
}

static int checkTestLosses(double* losses, int n) {
    int ret = 1;
    for (int i = 1; i < n; i++) {
        ret = ret && (losses[i - 1] > losses[i]);
    }
    return ret;
}

static void moveTestLosses(double* losses, int n) {
    for (int i = n - 1; i > 0; i--) {
        losses[i] = losses[i - 1];
    }
}

int modelTrain(model_schema_t* mem, int (*batchCallback)(model_schema_t* mem, int batchIndex, int step)) {
    int ret = 0;
    int printTrainProcess = mem->printTrainProcess;
    int trainBatchCount = ceilDivide(mem->trainCount, mem->batchSize);
    int testBatchCount = ceilDivide(mem->testCount, mem->batchSize);
    int accuracy;
    int total;
    double loss;
    double t;
    double* testLosses = initTestLosses(mem->lossCheckCount);
    int printMod = trainBatchCount / 10;
    if (printMod < 1) printMod = 1;

    for (int ep = 0; !ret && ep < mem->epoch; ep++) {
        ret = modelClearArguments(mem);
        t = 0;
        // 在训练集上训练
        for (int i = 0; !ret && i < trainBatchCount; i++) {
            int offset = i * mem->batchSize;
            int batchSize = minInt(mem->trainCount - offset, mem->batchSize);
            if (batchCallback != NULL) {
                ret = ret || (*batchCallback)(mem, i, 0);
            }
            ret = ret || modelRunBatch(mem, offset, batchSize);
            if (batchCallback != NULL) {
                ret = ret || (*batchCallback)(mem, i, 1);
            }
            ret = ret || modelTrainBatch(mem, offset, batchSize);
            if (batchCallback != NULL) {
                ret = ret || (*batchCallback)(mem, i, 2);
            }
            t++;
            ret = ret || modelApplyDweights(mem, offset, t);
            if (batchCallback != NULL) {
                ret = ret || (*batchCallback)(mem, i, 3);
            }
            ret = ret || modelFetchAccuracyRate(mem, &accuracy);
            ret = ret || modelFetchTotalCount(mem, &total);
            ret = ret || modelFetchLoss(mem, &loss);
            if ((i % printMod == 0) && printTrainProcess) {
                printf("[%d]当前训练集正确率 = %10.6lf%%, 损失 = %.6lf\n", ep, accuracy * 100.0 / total, loss * 1.0 / total);
            }
        }

        // 在测试集上测试
        modelClearAccuracy(mem);
        for (int i = 0; !ret && i < testBatchCount; i++) {
            int offset = mem->trainCount + i * mem->batchSize;
            int batchSize = minInt(mem->trainCount + mem->testCount - offset, mem->batchSize);
            ret = ret || modelRunBatch(mem, offset, batchSize);
        }
        ret = ret || modelFetchAccuracyRate(mem, &accuracy);
        ret = ret || modelFetchTotalCount(mem, &total);
        ret = ret || modelFetchLoss(mem, &loss);

        // 输出测试集上正确率
        if (!ret) {
            printf("[%d]当前测试集正确率 = %10.6lf%%, 损失 = %.6lf\n", ep, accuracy * 100.0 / total, loss * 1.0 / total);
            // 防止过拟合
            testLosses[0] = loss * 1.0 / total;
            if (checkTestLosses(testLosses, mem->lossCheckCount)) {
                printf("测试集损失上升，检测到过拟合，提前退出\n");
                break;
            }
            moveTestLosses(testLosses, mem->lossCheckCount);
        }

        // 学习率衰减
        mem->studyRate *= mem->attenuationRate;
    }
    if (testLosses) delete [] testLosses;
    return ret;
}
