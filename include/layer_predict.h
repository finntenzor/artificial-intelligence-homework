#ifndef LAYER_PREDICT_H
#define LAYER_PREDICT_H

#include "layer_schema.h"

int layerPredictInput(layer_schema_t* schema, int batchSize, unsigned char* input);
int layerPredictConvolution(layer_schema_t* schema, int batchSize);
int layerPredictPooling(layer_schema_t* schema, int batchSize);
int layerPredictDense(layer_schema_t* schema, int batchSize);
int layerPredictScale(layer_schema_t* schema, int batchSize);
int layerPredictRelu(layer_schema_t* schema, int batchSize);
int layerPredictTanh(layer_schema_t* schema, int batchSize);
int layerPredictOutput(layer_schema_t* schema, int batchSize, unsigned char* predict);

#endif // LAYER_PREDICT_H
