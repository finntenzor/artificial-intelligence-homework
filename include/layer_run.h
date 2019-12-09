#ifndef LAYER_RUN_H
#define LAYER_RUN_H

#include "layer_schema.h"
#include "layer_run_common.h"
#include "layer_predict.h"
#include "layer_train.h"
#include "layer_funcs.h"

int layerCheckOutput(layer_schema_t* schema, int batchSize, unsigned char* output, unsigned char* labels, int* pacc, int* ptot, double* ploss);

#endif // LAYER_RUN_H
