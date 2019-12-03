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
    int layerIndex; // ���±�
    int type; // ������

    // ���������С
    int inputWidth;
    int inputHeight;
    int inputDepth;
    int outputWidth;
    int outputHeight;
    int outputDepth;

    // �������� ���/�ػ�����
    int operationWidth;
    int operationHeight;
    int operationRowStep;
    int operationColStep;
    int operationRowBasis;
    int operationColBasis;

    // �ڴ�����С
    int predictInputSize;
    int predictOutputSize;
    int predictTempSize; // ��Ҫ�ֶ�ָ��
    int trainInputSize;
    int trainOutputSize;
    int trainTempSize; // ��Ҫ�ֶ�ָ��
    int weightsSize; // ��Ҫ�ֶ�ָ��
    int dweightsSize;

    // �Դ�ָ��
    double* predictInput; // �����ǰ�����룬����ǰһ������
    double* predictOutput; // �����ǰ�������������һ������
    double* predictTemp; // �����ǰ�������м����
    double* trainInput; // ����ķ��򴫲����룬���Ժ�һ������
    double* trainOutput; // ����ķ��򴫲����������ǰһ������
    double* trainTemp; // ����ķ��򴫲������м����
    double* weights; // �����ģ��Ȩ��
    double* dweights; // �����ģ��Ȩ�ر仯��
} layer_schema_t;

void layerInitSizes(layer_schema_t* schema, int batchSize);
void layerConcatInputSize(layer_schema_t* schema, layer_schema_t* lastSchema);

#endif // LAYER_SCHEMA_H
