#include <cuda.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <sys/time.h>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <random>

#define FULL_MASK 0xFFFFFFFF

#define cudaCheckError(code)                                               \
{                                                                          \
    if ((code) != cudaSuccess) {                                           \
        fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
        cudaGetErrorString(code));                                         \
    }                                                                      \
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

template <typename T, const int numBlocks, const int blockSize>
__global__ void deviceReductionKernel1 (const T* __restrict__ arr, const int arr_start, const int arr_end, T* __restrict__ accum, const int accum_start){
    T sum = 0;
    for(int i = threadIdx.x + blockIdx.x * blockSize + arr_start; i < arr_end; i += blockSize * numBlocks) sum += arr[i];

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0x0000FFFF, sum, 8);
    sum += __shfl_down_sync(0x000000FF, sum, 4);
    sum += __shfl_down_sync(0x0000000F, sum, 2);
    sum += __shfl_down_sync(0x00000003, sum, 1);

    extern __shared__ T sharedMem[];
    const int laneId = threadIdx.x & 31;
    const int warpId = (threadIdx.x >> 5);

    if (laneId == 0) sharedMem[warpId] = sum;
    __syncthreads();

    if (warpId == 0) {
        sum = sharedMem[laneId];

        if constexpr (blockSize >= 1024) sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        if constexpr (blockSize >= 512)  sum += __shfl_down_sync(0x0000FFFF, sum, 8);
        if constexpr (blockSize >= 256)  sum += __shfl_down_sync(0x000000FF, sum, 4);
        if constexpr (blockSize >= 128)  sum += __shfl_down_sync(0x0000000F, sum, 2);
        if constexpr (blockSize >= 64)   sum += __shfl_down_sync(0x00000003, sum, 1);

        if (laneId == 0) accum[blockIdx.x + accum_start] = sum;     
    }
}

__device__ __forceinline__ int2 operator+(const int2 &a, const int2 & b){
    return make_int2(a.x + b.x, a.y + b.y);
}

template <typename T, const int numBlocks, const int blockSize, const int power, const int mask>
__global__ void deviceReductionKernel3 (const int2* __restrict__ arr, const int arr_start, T* __restrict__ accum, const int accum_start){
    int2 reg = make_int2(0, 0);
    const int start = threadIdx.x + ((mask * (1 << power)) / (blockSize * numBlocks * 2)) * blockSize * blockIdx.x + (arr_start >> 1);

    #pragma unroll 
    for(int i = 0; i < ((mask * (1 << power)) / (blockSize * numBlocks * 2)); i++) reg = reg + arr[start + blockSize * i];

    T sum = reg.x + reg.y;
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0x0000FFFF, sum, 8);
    sum += __shfl_down_sync(0x000000FF, sum, 4);
    sum += __shfl_down_sync(0x0000000F, sum, 2);
    sum += __shfl_down_sync(0x00000003, sum, 1);

    extern __shared__ T sharedMem[];
    const int laneId = threadIdx.x & 31;
    const int warpId = (threadIdx.x >> 5);

    if (laneId == 0) sharedMem[warpId] = sum;
    __syncthreads();

    if (warpId == 0) {
        sum = sharedMem[laneId];

        if constexpr (blockSize >= 1024) sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        if constexpr (blockSize >= 512)  sum += __shfl_down_sync(0x0000FFFF, sum, 8);
        if constexpr (blockSize >= 256)  sum += __shfl_down_sync(0x000000FF, sum, 4);
        if constexpr (blockSize >= 128)  sum += __shfl_down_sync(0x0000000F, sum, 2);
        if constexpr (blockSize >= 64)   sum += __shfl_down_sync(0x00000003, sum, 1);

        if (laneId == 0) accum[blockIdx.x + accum_start] = sum;
    }
}

__device__ __forceinline__ int4 operator+(const int4 &a, const int4 & b){
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template <typename T, const int numBlocks, const int blockSize, const int power, const int mask>
__global__ void deviceReductionKernel4 (const int4* __restrict__ arr, T* __restrict__ accum){
    int4 reg = make_int4(0, 0, 0, 0);
    const int start = threadIdx.x + ((mask * (1 << power)) / (blockSize * numBlocks * 4)) * blockSize * blockIdx.x;

    #pragma unroll
    for(int i = 0; i < ((mask * (1 << power)) / (blockSize * numBlocks * 4)); i++) reg = reg + arr[start + blockSize * i];
    int sum = reg.x + reg.y + reg.z + reg.w;

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0x0000FFFF, sum, 8);
    sum += __shfl_down_sync(0x000000FF, sum, 4);
    sum += __shfl_down_sync(0x0000000F, sum, 2);
    sum += __shfl_down_sync(0x00000003, sum, 1);

    extern __shared__ T sharedMem[];
    const int laneId = threadIdx.x & 31;
    const int warpId = (threadIdx.x >> 5);

    if (laneId == 0) sharedMem[warpId] = sum;
    __syncthreads();

    if (warpId == 0) {
        sum = sharedMem[laneId];

        if constexpr (blockSize >= 1024) sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        if constexpr (blockSize >= 512)  sum += __shfl_down_sync(0x0000FFFF, sum, 8);
        if constexpr (blockSize >= 256)  sum += __shfl_down_sync(0x000000FF, sum, 4);
        if constexpr (blockSize >= 128)  sum += __shfl_down_sync(0x0000000F, sum, 2);
        if constexpr (blockSize >= 64)   sum += __shfl_down_sync(0x00000003, sum, 1);

        if (laneId == 0) accum[blockIdx.x + 1] = sum;
    }
}

template <typename T, const int blockSize>
__global__ void deviceReductionAccumulate(T* __restrict__ accum){
    T sum = accum[threadIdx.x + blockSize * blockIdx.x + 1];

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0x0000FFFF, sum, 8);
    sum += __shfl_down_sync(0x000000FF, sum, 4);
    sum += __shfl_down_sync(0x0000000F, sum, 2);
    sum += __shfl_down_sync(0x00000003, sum, 1);

    extern __shared__ T sharedMem[];
    const int laneId = threadIdx.x & 31;
    const int warpId = (threadIdx.x >> 5);

    if (laneId == 0) sharedMem[warpId] = sum;
    __syncthreads();

    if (warpId == 0) {
        sum = sharedMem[laneId];

        if constexpr (blockSize >= 1024) sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        if constexpr (blockSize >= 512)  sum += __shfl_down_sync(0x0000FFFF, sum, 8);
        if constexpr (blockSize >= 256)  sum += __shfl_down_sync(0x000000FF, sum, 4);
        if constexpr (blockSize >= 128)  sum += __shfl_down_sync(0x0000000F, sum, 2);
        if constexpr (blockSize >= 64)   sum += __shfl_down_sync(0x00000003, sum, 1);

        if (laneId == 0) atomicAdd(accum, sum);
    }
}

template <typename T>
__global__ void copyKernel(const T* __restrict__ arr, const int arr_start, const int arr_end, T* __restrict__ accum, const int accum_start){
    accum[threadIdx.x + accum_start] = (arr_start + threadIdx.x < arr_end ? arr[arr_start + threadIdx.x] : 0);
}

template <typename T, const int blockSize>
__global__ void reduceSmallInput(const T* __restrict__ arr, const int arr_start, const int arr_end, T* __restrict__ accum, const int accum_start){
    accum[accum_start + threadIdx.x] = 0;
    for(int i = arr_start + threadIdx.x; i < arr_end; i += blockSize) accum[accum_start + threadIdx.x] += arr[i];
}

template <typename T>
T computeReduction(T* d_arr, const int n){

    const int _0To20  = n & 0x001FFFFF;
    const int _21To25 = n & 0x03E00000;
    const int _26To30 = n & 0x7C000000;

    const int numReduced = 128 * (_0To20 != 0) + 256 * (_21To25 != 0) + 256 * (_26To30 != 0);
    
    T* accum;
    cudaMalloc(&accum, (numReduced + 1) * sizeof(T));
    cudaMemset(accum, 0, sizeof(T));

    int arr_start = 0;
    int accum_start = 1;

    // Combining 26 to 30
    if(_26To30){
        constexpr int numBlocks = 256;
        const int mask = _26To30 >> 26;
        switch(mask){
            case(0x01): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x01> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x02): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x02> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;    
            case(0x03): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x03> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x04): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x04> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x05): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x05> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x06): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x06> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x07): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x07> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x08): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x08> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x09): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x09> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x0A): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x0A> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;    
            case(0x0B): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x0B> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x0C): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x0C> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x0D): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x0D> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x0E): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x0E> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x0F): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x0F> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x10): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x10> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x11): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x11> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;    
            case(0x12): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x12> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x13): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x13> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x14): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x14> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x15): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x15> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x16): deviceReductionKernel4 <T, numBlocks, 256, 26, 0x16> <<< numBlocks, 256, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x17): deviceReductionKernel4 <T, numBlocks, 512, 26, 0x17> <<< numBlocks, 512, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x18): deviceReductionKernel4 <T, numBlocks, 512, 26, 0x18> <<< numBlocks, 512, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x19): deviceReductionKernel4 <T, numBlocks, 512, 26, 0x19> <<< numBlocks, 512, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;    
            case(0x1A): deviceReductionKernel4 <T, numBlocks, 512, 26, 0x1A> <<< numBlocks, 512, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x1B): deviceReductionKernel4 <T, numBlocks, 512, 26, 0x1B> <<< numBlocks, 512, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x1C): deviceReductionKernel4 <T, numBlocks, 512, 26, 0x1C> <<< numBlocks, 512, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x1D): deviceReductionKernel4 <T, numBlocks, 512, 26, 0x1D> <<< numBlocks, 512, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x1E): deviceReductionKernel4 <T, numBlocks, 512, 26, 0x1E> <<< numBlocks, 512, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
            case(0x1F): deviceReductionKernel4 <T, numBlocks, 512, 26, 0x1F> <<< numBlocks, 512, 32 * sizeof(T) >>> ((int4*)d_arr, accum);
                         break;
        }
        arr_start += _26To30;
        accum_start += numBlocks;
    }
 gpuErrchk( cudaPeekAtLastError() );
    // Combining 21 to 25
    if(_21To25){
        #define numBlocks 256 
        #define numThreads 256
        
        const int mask = _21To25 >> 21;
        switch(mask){
            case(0x01): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x01> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x02): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x02> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x03): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x03> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x04): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x04> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x05): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x05> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x06): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x06> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x07): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x07> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x08): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x08> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x09): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x09> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x0A): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x0A> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x0B): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x0B> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x0C): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x0C> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x0D): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x0D> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x0E): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x0E> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x0F): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x0F> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x10): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x10> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x11): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x11> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x12): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x12> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x13): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x13> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x14): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x14> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x15): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x15> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x16): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x16> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x17): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x17> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x18): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x18> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x19): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x19> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x1A): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x1A> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x1B): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x1B> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x1C): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x1C> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x1D): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x1D> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x1E): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x1E> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
            case(0x1F): deviceReductionKernel3 <T, numBlocks, numThreads, 21, 0x1F> <<< numBlocks, numThreads, 32 * sizeof(T) >>> ((int2*)d_arr, arr_start, accum, accum_start);
                         break;
        }
        arr_start += _21To25;
        accum_start += numBlocks;

        #undef numBlocks
        #undef numThreads
    }


    if(_0To20){
        #define numBlocks  128
        #define numThreads  256
        
        // the kernel has conditional checks
        if(_0To20 <= numBlocks) copyKernel <T> <<< 1, numBlocks >>> (d_arr, arr_start, arr_start + _0To20, accum, accum_start);
        else if(_0To20 <= numThreads * numBlocks) reduceSmallInput <T, numBlocks> <<< 1, numBlocks >>> (d_arr, arr_start, arr_start + _0To20, accum, accum_start);
        else deviceReductionKernel1 <T, numBlocks, numThreads> <<< numBlocks, numThreads, 32 * sizeof(T) >>> (d_arr, arr_start, arr_start + _0To20, accum, accum_start);

        arr_start += _0To20;
        accum_start += numBlocks;
        
        #undef numBlocks
        #undef numThreads
    } 

    cudaCheckError(cudaDeviceSynchronize());
    deviceReductionAccumulate <T, 128> <<< numReduced / 128 , 128, 32 * sizeof(T)>>> (accum);

    T h_result;
    cudaMemcpy(&h_result, accum, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(accum);

    return h_result;
}

int NUM_EXPERIMENTS;
template <typename T>
void generate_random_array(int n, T min_val, T max_val, T* arr) {
    // Create a random number generator
    std::random_device rd; // Seed source
    std::mt19937 gen(rd()); // Mersenne Twister engine

    // Select appropriate distribution based on type
    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(min_val, max_val);
        for (int i = 0; i < n; ++i) {
            arr[i] = dist(gen);
        }
    } else if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (int i = 0; i < n; ++i) {
            arr[i] = dist(gen);
        }
    } else {
        static_assert(std::is_arithmetic<T>::value, "Type must be numeric.");
    }
}


template <typename T>
T* generateRandomArrays(const int n){
    // allocate memory
    T* arr = new T[n];

    // initialize array
    if(n <= 1e6){
        generate_random_array <T> (n, 1, 1000, arr);
    }else if(n <= 1e7){
        generate_random_array <T> (n, 1, 100, arr);
    }else if(n <= 1e8){
        generate_random_array <T> (n, 1, 10, arr);
    }else if(n <= 1e9){
        generate_random_array <T> (n, 1, 2, arr);
    }else if(n <= 2*1e9){
        generate_random_array <T> (n, 1, 1, arr);
    }
    
    return arr;
}

template <typename T> 
T thrustReduction(int n, thrust::device_vector <T>& d_arr){
    return thrust::reduce(d_arr.begin(), d_arr.end(), 0, thrust::plus <T> ());
}

double calculateStandardDeviation(const std::vector<double>& data) {
    int n = data.size();
    if (n == 0) return 0.0; // Handle empty array case

    // Step 1: Calculate mean
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;

    // Step 2: Calculate variance
    double variance = 0.0;
    for (double x : data) {
        variance += (x - mean) * (x - mean);
    }
    variance /= n;

    // Step 3: Return standard deviation
    return std::sqrt(variance);
}


template <typename T>
void experiment(int low, int high){
    std::random_device rd; // Seed
    std::mt19937 gen(rd()); // Mersenne Twister generator
    std::uniform_int_distribution<> dis(low, high);

    FILE* fp = fopen("execution_times/comparision.csv", "w");
    if (!fp) {
        std::cerr << "Error: Could not open file for writing.\n";
        return;
    }
    printf("inputSize,thrustMedianTime,thrustMeanTime,thrustStandardDeviation,thrustSum,myReductionMedianTime,myReductionMeanTime,myReductionStandardDeviation,myReductionSum,myReductionCorrectness\n");
    fprintf(fp, "inputSize,thrustMedianTime,thrustMeanTime,thrustStandardDeviation,thrustSum,myReductionMedianTime,myReductionMeanTime,myReductionStandardDeviation,myReductionSum,myReductionCorrectness\n");


    int n = 1;
    while(n <= 2*1e9){
        T *h_arr = generateRandomArrays <T> (n);

        double thrustMedianTime, thrustMeanTime, thrustStandardDeviation;
        T thrustSum;
        {
            thrust::device_vector <T> d_arr(h_arr, h_arr + n);
        
            T sum;
            std::vector <double> execution_times;
            std::set <T> all_sums;
            for(int exp = 0; exp < NUM_EXPERIMENTS; exp++){
                double start_time = rtclock();

                sum = thrustReduction <T> (n, d_arr);
            
                double end_time = rtclock();
                double time_consumed = end_time - start_time;
                
                thrustSum = sum;
                
                all_sums.insert(sum);
                execution_times.push_back(time_consumed * 1e3);
            }
            if(all_sums.size() > 1){
                std::cout << "Thrust is giving multiple sums for same array\n";
                return;
            }
            sort(execution_times.begin(), execution_times.end());

            double medianTime;
            if(NUM_EXPERIMENTS % 2 == 1) medianTime = execution_times[NUM_EXPERIMENTS/2];
            else medianTime = (execution_times[NUM_EXPERIMENTS/2] + execution_times[NUM_EXPERIMENTS/2 - 1]) / 2;

            double meanTime = std::accumulate(execution_times.begin(), execution_times.end(), 0.0) / NUM_EXPERIMENTS;
            thrustMedianTime = medianTime;
            thrustMeanTime = meanTime;
            thrustStandardDeviation = calculateStandardDeviation(execution_times); 
        }

        int myReductionSum = 0;
        bool myReductionCorrectness = true;
        double myReductionMedianTime, myReductionMeanTime, myReductionStandardDeviation;
        {
            T *d_arr;
            cudaMalloc(&d_arr, n * sizeof(T));
            cudaMemcpy(d_arr, h_arr, n * sizeof(T), cudaMemcpyHostToDevice);

            T sum;
            std::vector <double> execution_times;
            for(int exp = 0; exp < NUM_EXPERIMENTS; exp++){
                double start_time = rtclock();

                sum = computeReduction <T> (d_arr, n);

                double end_time = rtclock();
                double time_consumed = end_time - start_time;

                if(sum != thrustSum){
                    myReductionCorrectness = false;
                }
                myReductionSum = sum;
                execution_times.push_back(time_consumed * 1e3);
            }
            sort(execution_times.begin(), execution_times.end());

            double medianTime;
            if(NUM_EXPERIMENTS % 2 == 1) medianTime = execution_times[NUM_EXPERIMENTS/2];
            else medianTime = (execution_times[NUM_EXPERIMENTS/2] + execution_times[NUM_EXPERIMENTS/2 - 1]) / 2;

            double meanTime = std::accumulate(execution_times.begin(), execution_times.end(), 0.0) / NUM_EXPERIMENTS;
            myReductionMedianTime = medianTime;
            myReductionMeanTime = meanTime;
            myReductionStandardDeviation = calculateStandardDeviation(execution_times); 

            cudaFree(d_arr);
        }
        fprintf(fp, "%d,%.12f,%.12f,%.12f,%d,%.12f,%.12f,%.12f,%d,%d\n", n, thrustMedianTime, thrustMeanTime, thrustStandardDeviation, thrustSum, myReductionMedianTime, myReductionMeanTime, myReductionStandardDeviation, myReductionSum, myReductionCorrectness);
        printf("%d,%.12f,%.12f,%.12f,%d,%.12f,%.12f,%.12f,%d,%d\n", n, thrustMedianTime, thrustMeanTime, thrustStandardDeviation, thrustSum, myReductionMedianTime, myReductionMeanTime, myReductionStandardDeviation, myReductionSum, myReductionCorrectness);

        delete[] h_arr;
        n += dis(gen);
    }
    fclose(fp);
}

int main(int argc, char* argv[]){
    if(argc != 4){
        printf("Expected interval for random difference, number of experiments as command line arguments\n");
        return 1;
    }

    int low = atoi(argv[1]);
    int high = atoi(argv[2]);
    NUM_EXPERIMENTS = atoi(argv[3]);
    experiment <int> (low, high);
}
