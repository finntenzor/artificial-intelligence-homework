#ifndef LAYER_SCHEMA_H
#define LAYER_SCHEMA_H

const int LAYER_TYPE_INPUT = 1;
const int LAYER_TYPE_CONVOLUTION = 2;
const int LAYER_TYPE_POOLING = 3;
const int LAYER_TYPE_DENSE = 4;
const int LAYER_TYPE_SCALE = 5;
const int LAYER_TYPE_RELU = 6;
const int LAYER_TYPE_OUTPUT = 7;

typedef struct layer_schema {
    int layerIndex; // 层下标
    int type; // 层类型

    // 输入输出大小
    int inputWidth;
    int inputHeight;
    int inputDepth;
    int outputWidth;
    int outputHeight;
    int outputDepth;

    // 操作参数 卷积/池化参数
    int operationWidth;
    int operationHeight;
    int operationRowStep;
    int operationColStep;
    int operationRowBasis;
    int operationColBasis;

    // 内存分配大小
    int predictInputSize;
    int predictOutputSize;
    int predictTempSize; // 需要手动指定
    int trainInputSize;
    int trainOutputSize;
    int trainTempSize; // 需要手动指定
    int weightsSize; // 需要手动指定
    int dweightsSize;

    // 显存指针
    double* predictInput; // 本层的前馈输入，来自前一层的输出
    double* predictOutput; // 本层的前馈输出，供给后一层输入
    double* predictTemp; // 本层的前馈计算中间变量
    double* trainInput; // 本层的反向传播输入，来自后一层的输出
    double* trainOutput; // 本层的反向传播输出，供给前一层输入
    double* trainTemp; // 本层的反向传播计算中间变量
    double* weights; // 本层的模型权重
    double* dweights; // 本层的模型权重变化量
} layer_schema_t;

void layerInitSizes(layer_schema_t* schema, int batchSize);
void layerConcatInputSize(layer_schema_t* schema, layer_schema_t* lastSchema);

#endif // LAYER_SCHEMA_H
