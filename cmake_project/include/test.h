#ifndef TEST_H_
#define TEST_H_

#include <vector>
#include <cuda.h>
#include <curand.h>
#include <assert.h>
#include <cufft.h>
#include <sstream>
#include <string>
#include "shorder.hpp"
#include "shproduct.h"
#include "shorder.hpp"
#include "select_size.hpp"

static const int N = select_size(n);
const int blocksize = 32;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
void initGamma();
void releaseGamma();
void traditional_method_gpu(float* A, float* B, float* C, int num);
__global__ void shprod_conventional(float* A, float* B, float* C);
void shprod_many(float* A, float* B, float* C, int num,
    cufftComplex *pool0, cufftComplex *pool1, cufftComplex *pool2);

#endif