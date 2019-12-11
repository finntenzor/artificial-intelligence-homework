/**
 * model_memory.cpp
 */

#include "model_memory.h"
#include "cuda.h"

#ifdef DEBUG
void debugModelCheckSchema(model_schema_t* mem) {
    printf("Model:\n");
    printf("input = %p\n", mem->input);
    printf("labels = %p\n", mem->labels);
    printf("output = %p\n", mem->output);
    printf("predictValues = %p\n", mem->predictValues);
    printf("predictTemps = %p\n", mem->predictTemps);
    printf("trainValues = %p\n", mem->trainValues);
    printf("trainTemps = %p\n", mem->trainTemps);
    printf("weights = %p\n", mem->weights);
    printf("dweights = %p\n", mem->dweights);
    printf("accuracyRate = %p\n", mem->accuracyRate);
    for (int i = 0; i < mem->schemaCount; i++) {
        printf("Layer %d:\n", i);
        printf("predictInput = %p\n", mem->schemas[i].predictInput);
        printf("predictOutput = %p\n", mem->schemas[i].predictOutput);
        printf("predictTemp = %p\n", mem->schemas[i].predictTemp);
        printf("trainInput = %p\n", mem->schemas[i].trainInput);
        printf("trainOutput = %p\n", mem->schemas[i].trainOutput);
        printf("trainTemp = %p\n", mem->schemas[i].trainTemp);
        printf("weights = %p\n", mem->schemas[i].weights);
        printf("dweights = %p\n", mem->schemas[i].dweights);
    }
}
#endif

int modelGetInputBlockSize(model_schema_t* mem) {
    layer_schema_t* inputSchema = &mem->schemas[0];
    return inputSchema->outputDepth * inputSchema->outputWidth * inputSchema->outputHeight;
}

int modelGetLabelsBlockSize(model_schema_t* mem) {
    return 1;
}

int modelGetOutputBlockSize(model_schema_t* mem) {
    return 1;
}

int modelGetInputSize(model_schema_t* mem) {
    return mem->memoryCount * modelGetInputBlockSize(mem);
}

int modelGetLabelsSize(model_schema_t* mem) {
    return mem->memoryCount * modelGetLabelsBlockSize(mem);
}

int modelGetOutputSize(model_schema_t* mem) {
    return mem->memoryCount * modelGetOutputBlockSize(mem);
}

int modelGetPredictValuesSize(model_schema_t* mem) {
    int size = 0;
    for (int i = 0; i < mem->schemaCount; i++) {
        size += mem->schemas[i].predictOutputSize;
    }
    return size;
}

int modelGetPredictTempSize(model_schema_t* mem) {
    int size = 0;
    for (int i = 0; i < mem->schemaCount; i++) {
        size += mem->schemas[i].predictTempSize;
    }
    return size;
}

int modelGetTrainValuesSize(model_schema_t* mem) {
    int size = 0;
    for (int i = 0; i < mem->schemaCount; i++) {
        size += mem->schemas[i].trainOutputSize;
    }
    return size;
}

int modelGetTrainTempSize(model_schema_t* mem) {
    int size = 0;
    for (int i = 0; i < mem->schemaCount; i++) {
        size += mem->schemas[i].trainTempSize;
    }
    return size;
}

int modelGetWeightsSize(model_schema_t* mem) {
    int size = 0;
    for (int i = 0; i < mem->schemaCount; i++) {
        size += mem->schemas[i].weightsSize;
    }
    return size;
}

int modelGetDweightsSize(model_schema_t* mem) {
    int size = 0;
    for (int i = 0; i < mem->schemaCount; i++) {
        size += mem->schemas[i].dweightsSize;
    }
    return size;
}

int modelGetTotalMemoryUsed(model_schema_t* mem) {
    int ucharCount = 0;
    int doubleCount = 0;

    ucharCount += modelGetInputSize(mem);
    ucharCount += modelGetLabelsSize(mem);
    ucharCount += modelGetOutputSize(mem);

    doubleCount += modelGetPredictValuesSize(mem);
    doubleCount += modelGetPredictTempSize(mem);
    doubleCount += modelGetTrainValuesSize(mem);
    doubleCount += modelGetTrainTempSize(mem);
    doubleCount += modelGetWeightsSize(mem);
    doubleCount += modelGetDweightsSize(mem);
    doubleCount += 1;

    return ucharCount * sizeof(unsigned char) + doubleCount * sizeof(double);
}

void modelInitLayerPointers(model_schema_t* mem) {
    layer_schema_t* schema = NULL;
    double* lastLayerPredictOutput = NULL;
    double* lastLayerTrainOutput = NULL;
    double* thisLayerPredictOutput = mem->predictValues;
    double* thisLayerPredictTemp = mem->predictTemps;
    double* thisLayerTrainOutput = mem->trainValues;
    double* thisLayerTrainTemp = mem->trainTemps;
    double* thisLayerWeights = mem->weights;
    double* thisLayerDWeights = mem->dweights;

    for (int i = 0; i < mem->schemaCount; i++) {
        schema = &(mem->schemas[i]);

        schema->predictInput = lastLayerPredictOutput;

        schema->predictOutput = thisLayerPredictOutput;
        schema->predictTemp = thisLayerPredictTemp;

        schema->weights = thisLayerWeights;
        schema->dweights = thisLayerDWeights;

        lastLayerPredictOutput = thisLayerPredictOutput;

        thisLayerPredictOutput += schema->predictOutputSize;
        thisLayerPredictTemp += schema->predictTempSize;

        thisLayerWeights += schema->weightsSize;
        thisLayerDWeights += schema->dweightsSize;
    }

    for (int i = mem->schemaCount - 1; i >= 0; i--) {
        schema = &(mem->schemas[i]);

        schema->trainInput = lastLayerTrainOutput;

        schema->trainOutput = thisLayerTrainOutput;
        schema->trainTemp = thisLayerTrainTemp;

        lastLayerTrainOutput = thisLayerTrainOutput;

        thisLayerTrainOutput += schema->trainOutputSize;
        thisLayerTrainTemp += schema->trainTempSize;
    }

    // debugModelCheckSchema(mem);
}

int modelAllocDeviceMemoryFor(model_schema_t* mem, void** dist, size_t size, const char* name) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaMalloc(dist, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "无法为 %s 分配内存，申请大小 = %zd Bytes (%.2lf MB), CUDA信息：%s\n", name, size, size / 1024.0 / 1024.0, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}

void modelClearPointers(model_schema_t* mem) {
    mem->input = NULL;
    mem->labels = NULL;
    mem->output = NULL;
    mem->predictValues = NULL;
    mem->predictTemps = NULL;
    mem->trainValues = NULL;
    mem->trainTemps = NULL;
    mem->weights = NULL;
    mem->dweights = NULL;
    mem->accuracyCount = NULL;
}

int modelAllocDeviceMemory(model_schema_t* mem) {
    cudaError_t cudaStatus = cudaSuccess;

    if (mem->schemaCount <= 0) {
        fprintf(stderr, "没有任何网络层, count = %d\n", mem->schemaCount);
        goto Error;
    }

    if (mem->schemas[0].type != LAYER_TYPE_INPUT) {
        fprintf(stderr, "第一层必须是输入层, type = %\n", mem->schemas[0].type);
        goto Error;
    }
    if (mem->schemas[mem->schemaCount - 1].type != LAYER_TYPE_OUTPUT) {
        fprintf(stderr, "最后一层必须是输出层, type = %d\n", mem->schemas[mem->schemaCount - 1].type);
        goto Error;
    }

    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->input, modelGetInputSize(mem) * sizeof(unsigned char), "输入集")) return 1;
    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->labels, modelGetLabelsSize(mem) * sizeof(unsigned char), "标签集")) return 1;
    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->output, modelGetOutputSize(mem) * sizeof(unsigned char), "输出集")) return 1;
    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->predictValues, modelGetPredictValuesSize(mem) * sizeof(double), "模型预测输入输出")) return 1;
    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->predictTemps, modelGetPredictTempSize(mem) * sizeof(double), "模型预测中间变量")) return 1;
    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->trainValues, modelGetTrainValuesSize(mem) * sizeof(double), "模型训练输入输出")) return 1;
    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->trainTemps, modelGetTrainTempSize(mem) * sizeof(double), "模型训练中间变量")) return 1;
    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->weights, modelGetWeightsSize(mem) * sizeof(double), "模型权重")) return 1;
    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->dweights, modelGetDweightsSize(mem) * sizeof(double), "模型权重变化量g")) return 1;
    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->mweights, modelGetDweightsSize(mem) * sizeof(double), "模型权重变化量m")) return 1;
    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->vweights, modelGetDweightsSize(mem) * sizeof(double), "模型权重变化量v")) return 1;
    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->accuracyCount, 1 * sizeof(int), "准确个数")) return 1;
    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->totalCount, 1 * sizeof(int), "训练个数")) return 1;
    if (modelAllocDeviceMemoryFor(mem, (void**)&mem->loss, 1 * sizeof(double), "损失")) return 1;

    modelInitLayerPointers(mem);

    return cudaStatus;

Error:
    modelFreeDeviceMemory(mem);
    return cudaStatus;
}

void modelFreeDeviceMemory(model_schema_t* mem) {
    cudaFree(mem->input);
    cudaFree(mem->labels);
    cudaFree(mem->output);
    cudaFree(mem->predictValues);
    cudaFree(mem->predictTemps);
    cudaFree(mem->trainValues);
    cudaFree(mem->trainTemps);
    cudaFree(mem->weights);
    cudaFree(mem->dweights);
    cudaFree(mem->accuracyCount);
    modelClearPointers(mem);
}

int modelCopyToDevice(void* dist, void* src, size_t size, const char* name) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaMemcpy(dist, src, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "无法将 %s 复制到显存, CUDA信息：%s\n", name, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}

int modelCopyFromDevice(void* dist, void* src, size_t size, const char* name) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaMemcpy(dist, src, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "无法从显存复制 %s 到内存, CUDA信息：%s\n", name, cudaGetErrorString(cudaStatus));
        return 1;
    }
    return 0;
}

int modelCopyInput(model_schema_t* mem, unsigned char* input) {
    return modelCopyToDevice(mem->input, input, modelGetInputSize(mem) * sizeof(unsigned char), "输入集");
}

int modelCopyTrainInput(model_schema_t* mem, unsigned char* train, unsigned char* test) {
    int ret = 0;
    int trainSize = mem->trainCount * modelGetInputBlockSize(mem) * sizeof(unsigned char);
    int testSize = mem->testCount * modelGetInputBlockSize(mem) * sizeof(unsigned char);
    ret = ret || modelCopyToDevice(mem->input, train, trainSize, "训练输入集");
    ret = ret || modelCopyToDevice(mem->input + trainSize, test, testSize, "测试输入集");
    return ret;
}

int modelCopyPredictInput(model_schema_t* mem, unsigned char* predict) {
    int predictSize = mem->predictCount * modelGetInputBlockSize(mem) * sizeof(unsigned char);
    return modelCopyToDevice(mem->input, predict, predictSize, "预测输入集");
}

int modelCopyLabels(model_schema_t* mem, unsigned char* labels) {
    return modelCopyToDevice(mem->labels, labels, modelGetLabelsSize(mem) * sizeof(unsigned char), "标签集");
}

int modelCopyTrainLabels(model_schema_t* mem, unsigned char* train, unsigned char* test) {
    int ret = 0;
    int trainSize = mem->trainCount * modelGetLabelsBlockSize(mem) * sizeof(unsigned char);
    int testSize = mem->testCount * modelGetLabelsBlockSize(mem) * sizeof(unsigned char);
    ret = ret || modelCopyToDevice(mem->labels, train, trainSize, "训练标签集");
    ret = ret || modelCopyToDevice(mem->labels + trainSize, test, testSize, "测试标签集");
    return ret;
}

int modelCopyOutput(model_schema_t* mem, unsigned char* output) {
    return modelCopyFromDevice(output, mem->output, modelGetOutputSize(mem) * sizeof(unsigned char), "输出集");
}

int modelCopyPredictOutput(model_schema_t* mem, unsigned char* output) {
    return modelCopyFromDevice(output, mem->output, mem->predictCount * modelGetOutputBlockSize(mem) * sizeof(unsigned char), "预测输出集");
}

int modelCopyToWeights(model_schema_t* mem, double* weights) {
    return modelCopyToDevice(mem->weights, weights, modelGetWeightsSize(mem) * sizeof(double), "模型权重");
}

int modelCopyFromWeights(model_schema_t* mem, double* weights) {
    return modelCopyFromDevice(weights, mem->weights, modelGetWeightsSize(mem) * sizeof(double), "模型权重");
}

int modelCopyFromDweights(model_schema_t* mem, double* weights) {
    return modelCopyFromDevice(weights, mem->weights, modelGetDweightsSize(mem) * sizeof(double), "模型权重变化量");
}
