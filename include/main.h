#ifndef MAIN_H
#define MAIN_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cli.h"
#include "visual_funcs.h"
#include "read_file.h"
#include "config.h"
#include "layer.h"
#include "model.h"

typedef struct model_config {
    ModelFacadeBuilder* builder;
    char trainImage[1000];
    char trainLabel[1000];
    char testImage[1000];
    char testLabel[1000];
    char loadPath[1000];
    char savePath[1000];
    int epoch;
    int batchSize;
    int trainImageCount;
    int testImageCount;
    int predictImageCount;
    int predictOutputCount;
    double studyRate;
    double attenuationRate;
    int printMemoryUsed;
    int printTrainProcess;
    int printPredictOutput;
    int printPredictAccuracyRate;
    int printModelSchema;
    int lossCheckCount;
} model_config_t;

int showDevices();
void initConfig(model_config_t* config, ModelFacadeBuilder* builder);
int readConfig(model_config_t* config, const char* configPath);
int beforeModule(void* dist, const char* module);
int readLayer(void* dist, const char* layerName, const int n, const int argv[]);

#endif // MAIN_H
