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
int modelGetBatchCount(model_schema_t* mem);
int modelRunBatch(model_schema_t* mem, int offset);
int modelFetchAccuracyRate(model_schema_t* mem, double* acc);
int modelFetchLoss(model_schema_t* mem, double* loss);
int modelPredict(model_schema_t* mem);
int modelClearDweights(model_schema_t* mem);
int modelTrainBatch(model_schema_t* mem, int offset);
int modelApplyDweights(model_schema_t* mem, int offset);
int modelTrain(model_schema_t* mem, int (*batchCallback)(model_schema_t* mem, int batchIndex, int step), int printTrainProcess);

#endif // MODEL_RUN_H
