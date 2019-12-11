#ifndef MODEL_MEMORY_H
#define MODEL_MEMORY_H

#include "model_schema.h"

int modelGetInputBlockSize(model_schema_t* mem);
int modelGetLabelsBlockSize(model_schema_t* mem);
int modelGetOutputBlockSize(model_schema_t* mem);
int modelGetInputSize(model_schema_t* mem);
int modelGetLabelsSize(model_schema_t* mem);
int modelGetOutputSize(model_schema_t* mem);
int modelGetPredictValuesSize(model_schema_t* mem);
int modelGetPredictTempSize(model_schema_t* mem);
int modelGetTrainValuesSize(model_schema_t* mem);
int modelGetTrainTempSize(model_schema_t* mem);
int modelGetWeightsSize(model_schema_t* mem);
int modelGetDweightsSize(model_schema_t* mem);

int modelGetTotalMemoryUsed(model_schema_t* mem);

void modelInitLayerPointers(model_schema_t* mem);
int modelAllocDeviceMemoryFor(model_schema_t* mem, void** dist, size_t size, const char* name);
int modelAllocDeviceMemory(model_schema_t* mem);
void modelFreeDeviceMemory(model_schema_t* mem);

int modelCopyToDevice(void* dist, void* src, size_t size, const char* name);
int modelCopyFromDevice(void* dist, void* src, size_t size, const char* name);
int modelCopyInput(model_schema_t* mem, unsigned char* input);
int modelCopyTrainInput(model_schema_t* mem, unsigned char* train, unsigned char* test);
int modelCopyPredictInput(model_schema_t* mem, unsigned char* predict);
int modelCopyLabels(model_schema_t* mem, unsigned char* labels);
int modelCopyTrainLabels(model_schema_t* mem, unsigned char* train, unsigned char* test);
int modelCopyOutput(model_schema_t* mem, unsigned char* output);
int modelCopyPredictOutput(model_schema_t* mem, unsigned char* output);
int modelCopyToWeights(model_schema_t* mem, double* weights);
int modelCopyFromWeights(model_schema_t* mem, double* weights);
int modelCopyFromDweights(model_schema_t* mem, double* weights);

#endif // MODEL_MEMORY_H
