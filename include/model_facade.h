#ifndef MODEL_FACADE_H
#define MODEL_FACADE_H

#include "layer_schema.h"
#include "model_schema.h"
#include "model_memory.h"
#include "model_run.h"

class ModelFacadeBuilder;

class ModelFacade: protected model_schema_t {
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
    int saveModel(const char* filepath);
    int loadModel(const char* filepath);
    int predict(unsigned char* input, unsigned char* output);
    double getAccuracyRate();
    int train(unsigned char* trainInput, unsigned char* trainLabels, unsigned char* testInput, unsigned char* testLabels);
    void setTrainListener(int (*trainListener)(model_schema_t* mem, int batchIndex, int step));
    layer_schema_t* layerAt(int index);
    void setStudyRate(double studyRate);
    void setAttenuationRate(double attenuationRate);
    void setEpoch(int epoch);
    void setPrintTrainProcess(int printTrainProcess);
    void setLossCheckCount(int lossCheckCount);
    void printSchema();
};

#endif // MODEL_FACADE_H
