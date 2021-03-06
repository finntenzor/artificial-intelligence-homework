/**
 * model_facade_builder.cpp
 */

#include "model_facade_builder.h"
#include <cstdio>

ModelFacadeBuilder::ModelFacadeBuilder() {
    batchSize = 0;
    memoryCount = 0;
    trainCount = 0;
    testCount = 0;
    predictCount = 0;
}
ModelFacadeBuilder::~ModelFacadeBuilder() {
    batchSize = 0;
    memoryCount = 0;
    trainCount = 0;
    testCount = 0;
    predictCount = 0;
}

void ModelFacadeBuilder::setMemory(int memoryCount, int trainCount, int testCount, int predictCount, int batchSize) {
    this->batchSize = batchSize;
    this->memoryCount = memoryCount;
    this->trainCount = trainCount;
    this->testCount = testCount;
    this->predictCount = predictCount;
}

void ModelFacadeBuilder::input(int width, int height) {
    layer_schema_t schema;
    int index = layers.size();

    schema.layerIndex = index;
    schema.type = LAYER_TYPE_INPUT;

    schema.inputWidth = width;
    schema.inputHeight = height;
    schema.inputDepth = 1;
    schema.outputWidth = width;
    schema.outputHeight = height;
    schema.outputDepth = 1;

    schema.operationWidth = 0;
    schema.operationHeight = 0;
    schema.operationRowStep = 0;
    schema.operationColStep = 0;
    schema.operationRowBasis = 0;
    schema.operationColBasis = 0;

    schema.predictTempSize = 0;
    schema.trainTempSize = 0;
    schema.weightsSize = 0;
    layerInitSizes(&schema, batchSize);
    // 输入层输入不来自显存
    schema.predictInputSize = schema.trainOutputSize = 0;
    layers.push_back(schema);
}

void ModelFacadeBuilder::convolution(int channels, int kernelWidth, int kernelHeight, int rowStep, int colStep, int rowBasis, int colBasis, int outputWidth, int outputHeight) {
    layer_schema_t schema;
    int index = layers.size();

    schema.layerIndex = index;
    schema.type = LAYER_TYPE_CONVOLUTION;

    layerConcatInputSize(&schema, &layers[index - 1]);
    if (outputWidth == 0) outputWidth = (schema.inputWidth - colBasis - kernelWidth + colStep) / colStep;
    if (outputHeight == 0) outputHeight = (schema.inputHeight - rowBasis - kernelHeight + rowStep) / rowStep;

    schema.outputWidth = outputWidth;
    schema.outputHeight = outputHeight;
    schema.outputDepth = channels;

    schema.operationWidth = kernelWidth;
    schema.operationHeight = kernelHeight;
    schema.operationRowStep = rowStep;
    schema.operationColStep = colStep;
    schema.operationRowBasis = rowBasis;
    schema.operationColBasis = colBasis;

    schema.predictTempSize = 0;
    schema.trainTempSize = batchSize * (schema.outputDepth * schema.outputWidth * schema.outputHeight);
    schema.weightsSize = channels * (schema.inputDepth * kernelWidth * kernelHeight + 1);
    layerInitSizes(&schema, batchSize);
    layers.push_back(schema);
}

void ModelFacadeBuilder::convolution(int channels, int kernelSize, int step, int basis, int outputSize) {
    this->convolution(channels, kernelSize, kernelSize, step, step, basis, basis, outputSize, outputSize);
}

void ModelFacadeBuilder::pooling(int windowWidth, int windowHeight, int rowStep, int colStep, int rowBasis, int colBasis, int outputWidth, int outputHeight) {
    layer_schema_t schema;
    int index = layers.size();

    schema.layerIndex = index;
    schema.type = LAYER_TYPE_POOLING;

    layerConcatInputSize(&schema, &layers[index - 1]);
    if (outputWidth == 0) outputWidth = (schema.inputWidth - colBasis - windowWidth + colStep) / colStep;
    if (outputHeight == 0) outputHeight = (schema.inputHeight - rowBasis - windowHeight + rowStep) / rowStep;

    schema.outputWidth = outputWidth;
    schema.outputHeight = outputHeight;
    schema.outputDepth = layers[index - 1].outputDepth;

    schema.operationWidth = windowWidth;
    schema.operationHeight = windowHeight;
    schema.operationRowStep = rowStep;
    schema.operationColStep = colStep;
    schema.operationRowBasis = rowBasis;
    schema.operationColBasis = colBasis;

    schema.predictTempSize = batchSize * schema.outputDepth * outputWidth * outputHeight * (windowWidth * windowHeight);
    schema.trainTempSize = 0;
    schema.weightsSize = schema.inputDepth * schema.inputWidth * schema.inputHeight;
    layerInitSizes(&schema, batchSize);
    layers.push_back(schema);
}

void ModelFacadeBuilder::pooling(int windowSize, int step, int basis, int outputSize) {
    if (step == 0) step = windowSize;
    this->pooling(windowSize, windowSize, step, step, basis, basis, outputSize, outputSize);
}

void ModelFacadeBuilder::dense(int channels) {
    layer_schema_t schema;
    int index = layers.size();

    schema.layerIndex = index;
    schema.type = LAYER_TYPE_DENSE;

    layerConcatInputSize(&schema, &layers[index - 1]);
    schema.outputWidth = 1;
    schema.outputHeight = 1;
    schema.outputDepth = channels;

    schema.operationWidth = 0;
    schema.operationHeight = 0;
    schema.operationRowStep = 0;
    schema.operationColStep = 0;
    schema.operationRowBasis = 0;
    schema.operationColBasis = 0;

    schema.predictTempSize = 0;
    schema.trainTempSize = batchSize * channels;
    schema.weightsSize = channels * (schema.inputDepth * schema.inputWidth * schema.inputHeight + 1);
    layerInitSizes(&schema, batchSize);
    layers.push_back(schema);
}

void ModelFacadeBuilder::scale() {
    layer_schema_t schema;
    layer_schema_t* lastSchema;
    int index = layers.size();
    lastSchema = &layers[index - 1];

    schema.layerIndex = index;
    schema.type = LAYER_TYPE_SCALE;

    layerConcatInputSize(&schema, &layers[index - 1]);
    schema.outputWidth = lastSchema->outputWidth;
    schema.outputHeight = lastSchema->outputHeight;
    schema.outputDepth = lastSchema->outputDepth;

    schema.operationWidth = 0;
    schema.operationHeight = 0;
    schema.operationRowStep = 0;
    schema.operationColStep = 0;
    schema.operationRowBasis = 0;
    schema.operationColBasis = 0;

    schema.predictTempSize = batchSize * 1; // 每个block存一个最大值
    schema.trainTempSize = 0;
    schema.weightsSize = 0;
    layerInitSizes(&schema, batchSize);
    layers.push_back(schema);
}

void ModelFacadeBuilder::relu() {
    layer_schema_t schema;
    int index = layers.size();

    schema.layerIndex = index;
    schema.type = LAYER_TYPE_RELU;

    layerConcatInputSize(&schema, &layers[index - 1]);
    schema.outputWidth = schema.inputWidth;
    schema.outputHeight = schema.inputHeight;
    schema.outputDepth = schema.inputDepth;

    schema.operationWidth = 0;
    schema.operationHeight = 0;
    schema.operationRowStep = 0;
    schema.operationColStep = 0;
    schema.operationRowBasis = 0;
    schema.operationColBasis = 0;

    schema.predictTempSize = 0;
    schema.trainTempSize = 0;
    schema.weightsSize = 0;
    layerInitSizes(&schema, batchSize);
    layers.push_back(schema);
}

void ModelFacadeBuilder::tanh() {
    layer_schema_t schema;
    int index = layers.size();

    schema.layerIndex = index;
    schema.type = LAYER_TYPE_TANH;

    layerConcatInputSize(&schema, &layers[index - 1]);
    schema.outputWidth = schema.inputWidth;
    schema.outputHeight = schema.inputHeight;
    schema.outputDepth = schema.inputDepth;

    schema.operationWidth = 0;
    schema.operationHeight = 0;
    schema.operationRowStep = 0;
    schema.operationColStep = 0;
    schema.operationRowBasis = 0;
    schema.operationColBasis = 0;

    schema.predictTempSize = 0;
    schema.trainTempSize = 0;
    schema.weightsSize = 0;
    layerInitSizes(&schema, batchSize);
    layers.push_back(schema);
}

void ModelFacadeBuilder::output() {
    layer_schema_t schema;
    layer_schema_t* lastSchema;
    int index = layers.size();
    lastSchema = &layers[index - 1];
    int outputSize = lastSchema->outputDepth * lastSchema->outputHeight * lastSchema->outputWidth;

    schema.layerIndex = index;
    schema.type = LAYER_TYPE_OUTPUT;

    layerConcatInputSize(&schema, &layers[index - 1]);
    schema.outputWidth = lastSchema->outputWidth;
    schema.outputHeight = lastSchema->outputHeight;
    schema.outputDepth = lastSchema->outputDepth;

    schema.operationWidth = 0;
    schema.operationHeight = 0;
    schema.operationRowStep = 0;
    schema.operationColStep = 0;
    schema.operationRowBasis = 0;
    schema.operationColBasis = 0;

    schema.predictTempSize = batchSize * 1 // 10个x中的最大值
        + batchSize * 1 // 10个exp(x-a)的和
        + batchSize * outputSize // 所有的exp(x-a)
        + batchSize * outputSize; // 所有的y
    schema.trainTempSize = 0;
    schema.weightsSize = 0;
    layerInitSizes(&schema, batchSize);
    // 输出层不给出double输出，但是训练不使用
    schema.predictOutputSize = 0;
    schema.trainInputSize = 0;
    layers.push_back(schema);
}

int ModelFacadeBuilder::build(ModelFacade* model) {
    int n = (int)layers.size();
    model->schemaCount = n;
    model->memoryCount = memoryCount;
    model->trainCount = trainCount;
    model->testCount = testCount;
    model->predictCount = predictCount;
    model->batchSize = batchSize;
    model->schemas = new layer_schema_t[n];
    model->studyRate = 0.01; // 默认学习率

    // 显存指针自动赋值，不需要清空

    // 拷贝每一层的结构
    for (int i = 0; i < n; i++) {
        model->schemas[i] = layers[i];
    }

    // 此时每一层结构除指针外均已经正确，可以求得权重大小
    // 但是不能赋新数组，需要强制在train时自动生成
    model->hostWeights = NULL;

    // 分配完毕后，每一层的指针也已经就位
    return modelAllocDeviceMemory(model);
}
