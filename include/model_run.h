#ifndef MODEL_RUN_H
#define MODEL_RUN_H

#include "layer_run.h"
#include "model_schema.h"
#include "model_memory.h"

#ifndef CHECK
#include "cuda_runtime.h"
#endif

int modelIfErrorWithStatus(cudaError_t cudaStatus, const char* str);
int modelIfError(const char* str);
int modelCalcGridThreadCount(int total, int* block, int* thread);
int modelRunBatch(model_schema_t* mem, int offset, int batchSize);
int modelFetchAccuracyRate(model_schema_t* mem, int* acc);
int modelFetchTotalCount(model_schema_t* mem, int* totalCount);
int modelFetchLoss(model_schema_t* mem, double* loss);
int modelPredict(model_schema_t* mem);
int modelClearDweights(model_schema_t* mem);
int modelClearAccuracy(model_schema_t* mem);
int modelClearArguments(model_schema_t* mem);
int modelTrainBatch(model_schema_t* mem, int offset, int batchSize);
int modelApplyDweights(model_schema_t* mem, int offset);
int modelTrain(model_schema_t* mem, int (*batchCallback)(model_schema_t* mem, int batchIndex, int step));

#endif // MODEL_RUN_H
