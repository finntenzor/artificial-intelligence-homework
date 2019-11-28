#ifndef READ_FILE_H
#define READ_FILE_H

#include "visual_funcs.h"

class ImageSet {
public:
    unsigned int count;
    unsigned int width;
    unsigned int height;
    unsigned char* images;
    ImageSet();
    ~ImageSet();
    int read(const char* filepath);
    unsigned char* getImage(int index);
    void printImage(int index);
};

class LabelSet {
public:
    unsigned int count;
    unsigned char* labels;
    LabelSet();
    ~LabelSet();
    int read(const char* filepath);
    int getLabel(int index);
};

#endif // READ_FILE_H
