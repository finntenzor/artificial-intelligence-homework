#ifndef MODEL_SCHEMA_H
#define MODEL_SCHEMA_H

#include "layer_schema.h"

typedef struct model_schema {
    int schemaCount;
    int inputCount;
    int batchSize;
    layer_schema_t* schemas;
    unsigned char* input; // ���е����뼯
    unsigned char* labels; // ���еı�ǩ��
    unsigned char* output; // ģ�����
    double* predictValues; // ÿһ���Ԥ���������
    double* predictTemps; // ÿһ���Ԥ���м����
    double* trainValues; // ÿһ���ѵ���������
    double* trainTemps; // ÿһ���ѵ���м����
    double* weights; // ÿһ���ѵ��Ȩ��
    double* dweights; // ÿһ���ѵ��Ȩ�صĵ���
    double* mweights; // m
    double* vweights; // v
    int* accuracyCount; // ׼ȷ����
    int* totalCount; // ѵ������
    double* loss; // ��ʧ
    double studyRate; // ѧϰ��
    double attenuationRate; // ѧϰ��˥����<����>
    int roundCount; // ѧϰ������
} model_schema_t;

#endif // MODEL_SCHEMA_H
