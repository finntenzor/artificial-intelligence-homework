#ifndef MODEL_SCHEMA_H
#define MODEL_SCHEMA_H

#include "layer_schema.h"

typedef struct model_schema {
    int schemaCount;
    int inputCount;
    int batchSize;
    layer_schema_t* schemas;
    unsigned char* input; // 所有的输入集
    unsigned char* labels; // 所有的标签集
    unsigned char* output; // 模型输出
    double* predictValues; // 每一层的预测输入输出
    double* predictTemps; // 每一层的预测中间变量
    double* trainValues; // 每一层的训练输入输出
    double* trainTemps; // 每一层的训练中间变量
    double* weights; // 每一层的训练权重
    double* dweights; // 每一层的训练权重的导数
    double* mweights; // m
    double* vweights; // v
    int* accuracyCount; // 准确个数
    int* totalCount; // 训练个数
    double* loss; // 损失
    double studyRate; // 学习率
    double attenuationRate; // 学习率衰减率<废弃>
    int roundCount; // 学习多少轮
} model_schema_t;

#endif // MODEL_SCHEMA_H
