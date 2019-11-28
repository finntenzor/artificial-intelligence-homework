/**
 * read_file.cpp
 */

#include <cstdio>
#include <cstdlib>

#include "read_file.h"

ImageSet::ImageSet() {
    count = 0;
    width = 0;
    height = 0;
    images = NULL;
}

ImageSet::~ImageSet() {
    if (images != NULL) {
        free(images);
        images = NULL;
    }
}

static unsigned int readUnsignedIntFromFile(FILE* file) {
    char curr;
    unsigned int res = 0;
    for (int i = 0; i < 4; i++) {
        res <<= 8;
        curr = fgetc(file);
        res += (unsigned char)curr;
    }
    return res;
}

int ImageSet::read(const char* filepath) {
    FILE* f;

    f = fopen(filepath, "rb");
    if (f == NULL) {
        fprintf(stderr, "无法打开文件 %s\n", filepath);
        return -1;
    }

    readUnsignedIntFromFile(f);
    count = readUnsignedIntFromFile(f);
    width = readUnsignedIntFromFile(f);
    height = readUnsignedIntFromFile(f);

    int n = count * width * height;
    images = new unsigned char[n];
    fread(images, n, sizeof(unsigned char), f);

    fclose(f);
    return 0;
}

unsigned char* ImageSet::getImage(int index) {
    if (images == NULL) {
        return NULL;
    }
    if (index < 0 || index >= count) {
        return NULL;
    }
    return images + index * width * height;
}

void ImageSet::printImage(int index) {
    if (images == NULL) {
        printf("未打开图片\n");
    }
    if (index < 0 || index >= count) {
        printf("图片不存在\n");
    }
    unsigned char* image = getImage(index);
    printImageu(image, width, height);
}

LabelSet::LabelSet() {
    count = 0;
    labels = NULL;
}

LabelSet::~LabelSet() {
    if (labels != NULL) {
        free(labels);
        labels = NULL;
    }
}

int LabelSet::read(const char* filepath) {
    FILE* f = NULL;

    f = fopen(filepath, "rb");
    if (f == NULL) {
        fprintf(stderr, "无法打开文件 %s\n", filepath);
        return -1;
    }

    readUnsignedIntFromFile(f);
    count = readUnsignedIntFromFile(f);

    int n = count;
    labels = new unsigned char[n];
    fread(labels, n, sizeof(unsigned char), f);

    if (f) {
        fclose(f);
    }
    return 0;
}

int LabelSet::getLabel(int index) {
    if (labels == NULL) {
        printf("未打开标签\n");
    }
    if (index < 0 || index >= count) {
        printf("标签不存在\n");
    }
    unsigned char label = labels[index];
    return (int)label;
}
