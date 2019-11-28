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
    // �ڴ�ָ��
    double* predictInput; // �����ǰ�����룬����ǰһ������
    double* predictOutput; // �����ǰ�������������һ������
    double* predictTemp; // �����ǰ�������м����
    double* trainInput; // ����ķ��򴫲����룬���Ժ�һ������
    double* trainOutput; // ����ķ��򴫲����������ǰһ������
    double* trainTemp; // ����ķ��򴫲������м����
    double* weights; // �����ģ��Ȩ��
    double* dweights; // �����ģ��Ȩ�ر仯��
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
