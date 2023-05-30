#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp>

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
}

void clear2DKernel(float*** kernel, int size)
{
    for (int i = 0; i < size; ++i) {
        delete((*kernel)[i]);
    }
    delete[](*kernel);
    *kernel = nullptr;
}
void clear1DKernel(float** kernel, int size)
{
    delete(*kernel);
    *kernel = nullptr;
}

void printKernel(float** kernel, int size)
{
    for (int x = 0; x < size; ++x) {
        for (int y = 0; y < size; ++y) {
            std::cout << kernel[x][y] << ' ';
        }
        std::cout << std::endl;
    }
}
void print1DKernel(float* kernel, int size)
{
    for (int x = 0; x < size; ++x) {
        std::cout << kernel[x] << ' ';
    }
    std::cout << std::endl;
}

void horisontalBlur(float* kernel, float** src_image, float** res_image, int img_size, int kernel_size)
{
    int half_size = std::floor(kernel_size / 2.);
    for (int i = 0; i < img_size; ++i)
        for (int j = half_size; j < img_size - half_size; ++j) {
            for (int k_j = -half_size; k_j <= half_size; ++k_j) {
                res_image[i][j] += src_image[i][j + k_j] * kernel[k_j + half_size];
            }
        }
}

void verticalBlur(float* kernel, float** src_image, float** res_image, int img_size, int kernel_size)
{
    int half_size = std::floor(kernel_size / 2.);
    for (int j = half_size; j < img_size - half_size; ++j) {
        for (int k_j = -half_size; k_j <= half_size; ++k_j) {
            for (int i = 0; i < img_size; ++i)
                res_image[j][i] += src_image[j + k_j][i] * kernel[k_j + half_size];
        }
    }
}

void blur(float** kernel, float** src_image, float** res_image, int img_size, int kernel_size)
{
    int half_size = std::floor(kernel_size / 2.);
    for (int i = half_size; i < img_size - half_size; ++i)
        for (int j = half_size; j < img_size - half_size; ++j) {
            for (int k_i = -half_size; k_i <= half_size; ++k_i)
#pragma simd reduction(+:res_image[i][j])
                for (int k_j = -half_size; k_j <= half_size; ++k_j) {
                    res_image[i][j] += src_image[i + k_i][j + k_j] *
                        kernel[k_i + half_size][k_j + half_size];
                }
        }
}

int main(int argc, char** argv)
{
    clock_t start, stop;
    int k_size = 15, sigma = 1, i_size = 2000;
    float** src_img = createImage(i_size);
    //src_img[3][3] = 128;
    float** res_img = createImage(i_size);
    //resetImage(res_img, i_size);
    float** kernel = create2DKernel(k_size, sigma);
    std::cout << "Gaussian kernel:\n";
    printKernel(kernel, k_size);
    start = clock();
    blur(kernel, src_img, res_img, i_size, k_size);
    stop = clock();
    std::cout << "Elapsed time simple = " << ((double)(stop - start)) / CLOCKS_PER_SEC << " seconds" << std::endl;

    //printKernel(res_img, i_size);
    //resetImage(res_img, i_size);

    //float* krnl1D = create1DKernel(k_size, sigma);
    //print1DKernel(krnl1D, k_size);
    //start = clock();
    ////printKernel(src_img, i_size);
    //horisontalBlur(krnl1D, src_img, res_img, i_size, k_size);
    //resetImage(src_img, i_size);
    //verticalBlur(krnl1D, res_img, src_img, i_size, k_size);
    //stop = clock();
    //std::cout << "Elapsed time simple = " << ((double)(stop - start)) / CLOCKS_PER_SEC << " seconds" << std::endl;

    //printKernel(src_img, i_size);

    clear2DKernel(&kernel, k_size);
    clear2DKernel(&src_img, i_size);
    clear2DKernel(&res_img, i_size);
    //clear1DKernel(&krnl1D, k_size);
    /*cv::Mat image = cv::imread("C:\\Users\\k7912\\source\\repos\\GaussianBlur\\x64\\Release\\enna.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cout << "Image not found\n";
    }*/

    //try {
    //    //cv::imshow("name", image);
    //}
    //catch (cv::Exception ex) {
    //    std::cout << ex.what() << std::endl;
    //}

    return 0;
}
