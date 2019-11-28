#ifndef LAYER_FACADE_H
#define LAYER_FACADE_H

#include "layer_schema.h"
#include "layer_memory.h"
#include "visual_funcs.h"

class LayerFacade {
protected:
    void freeHeap();
    void allocHeap(layer_schema_t* schema);
public:
    layer_schema_t* schema;
    // 内存指针
    double* predictInput; // 本层的前馈输入，来自前一层的输出
    double* predictOutput; // 本层的前馈输出，供给后一层输入
    double* predictTemp; // 本层的前馈计算中间变量
    double* trainInput; // 本层的反向传播输入，来自后一层的输出
    double* trainOutput; // 本层的反向传播输出，供给前一层输入
    double* trainTemp; // 本层的反向传播计算中间变量
    double* weights; // 本层的模型权重
    double* dweights; // 本层的模型权重变化量
    LayerFacade();
    LayerFacade(layer_schema_t* schema);
    ~LayerFacade();
    void setLayerSchema(layer_schema_t* schema);
    void read();
    void printGeneralOutputImage(int blockIndex, int depth);
    void printGeneralOutputMatrix(int blockIndex, int depth);
    void printInputOutputImage(int index);
    void printFullConnectedArgs(int featureIndex, int depth);
    void printFullConnectedOutput(int beginBlock, int endBlock);
};

#endif // LAYER_FACADE_H
