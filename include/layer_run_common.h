#ifndef LAYER_RUN_COMMON_H
#define LAYER_RUN_COMMON_H

#include "layer_schema.h"

#ifndef CHECK
#include "device_launch_parameters.h"
#endif

__device__ int layerGetInputIndex(layer_schema_t* schema, int blockIndex, int channelIndex, int rowIndex, int colIndex);
__device__ int layerGetOutputIndex(layer_schema_t* schema, int blockIndex, int channelIndex, int rowIndex, int colIndex);
__device__ int layerGetCurrentOutputIndex(layer_schema_t* schema);

int layerIfError(int layerIndex);

#endif // LAYER_RUN_COMMON_H
