#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
// #include <opencv2/opencv.hpp>

#include <ctime>

float** createImage(int size)
{
    float** image = new float* [size];
    for (int i = 0; i < size; ++i) {
        image[i] = new float[size];
        for (int j = 0; j < size; ++j) {
            image[i][j] = 1;
        }
    }
    return image;
}

void resetImage(float** img, int size)
{
    for (int i = 0; i < size; ++i) {
        img[i] = new float[size];
        for (int j = 0; j < size; ++j) {
            img[i][j] = 0;
        }
    }
}

float* create1DKernel(int size, float sigma)
{
    float* kernel = new float[size];

    float r, s = 2.f * sigma * sigma;

    float sum = 0.;

    int half_size = std::floor(size / 2);

    for (int x = -half_size; x <= half_size; ++x) {
        kernel[x + half_size] = std::exp(-(x * x) / s) / (sigma * std::sqrt(M_PI * 2.f));
        sum += kernel[x + half_size];
    }
    for (int x = 0; x < size; ++x) {
        kernel[x] /= sum;
    }

    return kernel;
}

// Gausian blur kernel creation
float** create2DKernel(int size, float sigma)
{
    float** kernel = new float* [size];
    for (int i = 0; i < size; ++i) {
        kernel[i] = new float[size];
    }
    float r, s = 2.f * sigma * sigma;

    float sum = 0.;
    int half_size = std::floor(size / 2);
    for (int x = -half_size; x <= half_size; ++x) {
        for (int y = -half_size; y <= half_size; ++y) {
            r = std::sqrt(x * x + y * y);
            kernel[x + half_size][y + half_size] = std::exp(-(r * r) / s) / (M_PI * s);
            sum += kernel[x + half_size][y + half_size];
        }
    }
    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            kernel[x][y] /= sum;
        }
    }

    return kernel;
