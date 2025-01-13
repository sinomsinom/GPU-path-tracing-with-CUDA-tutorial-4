#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


/**
 * Macro for checking cuda errors following a cuda launch or api call
 */
#define cudaCheckError( e ) { cudaAssert((e), __FILE__, __LINE__); }

inline void cudaAssert(const cudaError_t e, const char* file, const int line ) {
    if (e != cudaSuccess) {
        printf("Cuda failure %s:%d: '%s'\n", file, line, cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}
