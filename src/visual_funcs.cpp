/**
 * visual_funcs.cpp
 */

#include <cstdio>
#include <cstdlib>

#include "visual_funcs.h"

void printImageu(unsigned char* image, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int k = i * width + j;
            printf("%c", image[k] > 0 ? '#' : ' ');
        }
        printf("\n");
    }
    printf("\n");
}

void printImagelf(double* image, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int k = i * width + j;
            printf("%c", image[k] > 0 ? '#' : ' ');
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrixu(unsigned char* matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int k = i * width + j;
            printf("%u ", matrix[k]);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrixlf(double* matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int k = i * width + j;
            printf("%.2lf ", matrix[k]);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrixlfk(double* matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int k = i * width + j;
            printf("%.2lf ", matrix[k] * 100.0);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrixlf4(double* matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int k = i * width + j;
            printf("%.4lf ", matrix[k]);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrixlf4k(double* matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int k = i * width + j;
            printf("%.4lf ", matrix[k] * 10000.0);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrixlf6(double* matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int k = i * width + j;
            printf("%.6lf ", matrix[k]);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrixlf6k(double* matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int k = i * width + j;
            printf("%.6lf ", matrix[k] * 1000000.0);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrixlf8(double* matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int k = i * width + j;
            printf("%.8lf ", matrix[k]);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrixlf8k(double* matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int k = i * width + j;
            printf("%.8lf ", matrix[k] * 100000000.0);
        }
        printf("\n");
    }
    printf("\n");
}
