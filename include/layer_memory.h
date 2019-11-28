#ifndef LAYER_MEMORY_H
#define LAYER_MEMORY_H

#include "layer_schema.h"

int layerCopyToDevice(layer_schema_t* schema, void* dist, void* src, size_t size, const char* name);
int layerCopyFromDevice(layer_schema_t* schema, void* dist, void* src, size_t size, const char* name);
int layerCopyFromDeviceDouble(layer_schema_t* schema, void* dist, void* src, int size, const char* name);
int layerCopyPredictInput(layer_schema_t* schema, double* input);
int layerCopyPredictOutput(layer_schema_t* schema, double* output);
int layerCopyPredictTemp(layer_schema_t* schema, double* temp);
int layerCopyTrainInput(layer_schema_t* schema, double* input);
int layerCopyTrainOutput(layer_schema_t* schema, double* output);
int layerCopyTrainTemp(layer_schema_t* schema, double* temp);
int layerCopyWeights(layer_schema_t* schema, double* weights);
int layerCopyDweights(layer_schema_t* schema, double* dweights);

#endif // LAYER_MEMORY_H
