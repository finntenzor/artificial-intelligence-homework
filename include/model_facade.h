#ifndef MODEL_FACADE_H
#define MODEL_FACADE_H

#include "layer_schema.h"
#include "model_schema.h"
#include "model_memory.h"
#include "model_run.h"

class ModelFacadeBuilder;

class ModelFacade: protected model_schema_t {
    int maxInputCount;
    double* hostWeights;
    double accuracyRate;
    int (*batchCallback)(model_schema_t* mem, int batchIndex, int step);
protected:
    void randomGenerateArgs();
public:
    ModelFacade();
    ~ModelFacade();
    friend class ModelFacadeBuilder;
    int getWeightsSize();
    int getTotalMemoryUsed();
    void saveModel(const char* filepath);
    void loadModel(const char* filepath);
    int predict(unsigned char* input, unsigned char* output, int totalCount = 0);
    double getAccuracyRate();
    int train(unsigned char* input, unsigned char* labels, int totalCount = 0);
    void setTrainListener(int (*trainListenr)(model_schema_t* mem, int batchIndex, int step));
    layer_schema_t* layerAt(int index);
    void setStudyRate(double studyRate);
};

#endif // MODEL_FACADE_H
