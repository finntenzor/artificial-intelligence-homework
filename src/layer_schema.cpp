/**
 * layer_schema.cpp
 */

#include "layer_schema.h"

void layerInitSizes(layer_schema_t* schema, int batchSize) {
    schema->predictInputSize = batchSize * schema->inputDepth * schema->inputHeight * schema->inputWidth;
    schema->predictOutputSize = batchSize * schema->outputDepth * schema->outputHeight * schema->outputWidth;
    schema->trainInputSize = batchSize * schema->outputDepth * schema->outputHeight * schema->outputWidth;
    schema->trainOutputSize = batchSize * schema->inputDepth * schema->inputHeight * schema->inputWidth;
    schema->dweightsSize = schema->weightsSize;
}

void layerConcatInputSize(layer_schema_t* schema, layer_schema_t* lastSchema) {
    schema->inputDepth = lastSchema->outputDepth;
    schema->inputHeight = lastSchema->outputHeight;
    schema->inputWidth = lastSchema->outputWidth;
}
