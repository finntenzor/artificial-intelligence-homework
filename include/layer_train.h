#ifndef LAYER_TRAIN_H
#define LAYER_TRAIN_H

#include "layer_schema.h"

int layerTrainOutput(layer_schema_t* schema, int batchSize, unsigned char* labels);
int layerTrainScale(layer_schema_t* schema, int batchSize);
int layerTrainDense(layer_schema_t* schema, int batchSize);

#endif // LAYER_TRAIN_H