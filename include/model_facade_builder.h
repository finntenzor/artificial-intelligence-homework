#ifndef MODEL_FACADE_BUILDER_H
#define MODEL_FACADE_BUILDER_H

#ifndef CHECK
#include <vector>
#endif

#include "layer_schema.h"
#include "model_schema.h"

#include "model_facade.h"
#include "model_memory.h"

class ModelFacadeBuilder {
public:
    ModelFacadeBuilder();
    ~ModelFacadeBuilder();
    int batchSize;
    int inputCount;
    std::vector<layer_schema_t> layers;
    void setMemory(int inputCount, int batchSize);
    void input(int width, int height);
    void convolution(int channels, int kernelWidth, int kernelHeight, int rowStep, int colStep, int rowBasis, int colBasis);
    void convolution(int channels, int kernelSize, int step, int basis = 0);
    void pooling(int windowWidth, int windowHeight, int rowStep, int colStep, int rowBasis, int colBasis);
    void pooling(int windowSize, int step = 0, int basis = 0);
    void dense(int channels);
    void scale();
    void output();
    void build(ModelFacade* model);
};

#endif // MODEL_FACADE_BUILDER_H
