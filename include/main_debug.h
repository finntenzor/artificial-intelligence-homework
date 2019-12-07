#ifndef MAIN_DEBUG
#define MAIN_DEBUG

#include <cstdio>
#include <cmath>
#include <cstring>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "visual_funcs.h"
#include "layer.h"
#include "model.h"

void printGeneralOutputImage(LayerFacade& layer, int blockIndex, int depth);
void printGeneralOutputMatrix(LayerFacade& layer, int blockIndex, int depth);
void printInputOutputImage(LayerFacade& layer, int index);
void printFullConnectedArgs(LayerFacade& layer, int featureIndex, int depth);
void printFullConnectedOutput(LayerFacade& layer, int beginBlock, int endBlock = -1);
void printPredictOutput(LayerFacade& layer, int depth = 0);
void printPredictInput(LayerFacade& layer, int depth = 0);
void printTrainOuput(LayerFacade& layer, int depth = 0);
void printWeights(LayerFacade& layer, int depth = 0);
void printDweights(LayerFacade& layer, int depth = 0);
void printConvolutionLayer(LayerFacade& layer);
void print3DArray(double* dist, int width, int height, int depth);
void printWholeConvolutionLayer(LayerFacade& layer, int m = 0);
void printOutputY(LayerFacade& layer, int batchSize);

#endif // MAIN_DEBUG
