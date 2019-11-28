#ifndef LAYER_FUNCS_H
#define LAYER_FUNCS_H

#include "layer_schema.h"

void layerOutputGetTempPointers(layer_schema_t* schema, int batchSize, double** expx, double** expxSum, double** y);

#endif // LAYER_FUNCS_H
