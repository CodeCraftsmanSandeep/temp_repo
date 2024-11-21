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

template <typename T, const int numBlocksPower, const int blockSizePower>
__global__ void deviceReductionKernel(const T* __restrict__ arr, const int n, T* __restrict__ accum){
    T sum = 0;
    const unsigned int workPerBlock = ((n >> (numBlocksPower + blockSizePower)) << blockSizePower);

    if(blockIdx.x == (1 << numBlocksPower)){
        // last block
        // 2^0 to 2^(numBlocksPower + blockSizePower - 1)
        for(int i = (workPerBlock << numBlocksPower) + threadIdx.x; i < n; i += (1 << blockSizePower)) sum += arr[i];
    }else{
        // 2^(numBlocksPower + blockSizePower) to 2^30
        unsigned int start = threadIdx.x + workPerBlock * blockIdx.x;

        // 2^(numBlocksPower + blockSizePower + 10) to 2^30
        for(unsigned int i = 0; i < (n >> (numBlocksPower + blockSizePower + 10)); i++){
            // unrolling 1024 times
            #pragma unroll
            for(unsigned int iter = 0; iter < (1 << 10); iter++){
                sum += arr[start];
                start += (1 << blockSizePower);
            }
        }

        // 2^(numBlocksPower + blockSizePower + 6) to 2^(numBlocksPower + blockSizePower + 9)
        constexpr int maskMiddleChunck = (~((1 << (numBlocksPower + blockSizePower + 6)) - 1)) & ((1 << (numBlocksPower + blockSizePower + 9 + 1)) - 1);
        for(unsigned int i = 0; i < ((n & maskMiddleChunck) >> (numBlocksPower + blockSizePower + 6)); i++){
            // unrolling 64 times
            #pragma unroll
            for(unsigned int iter = 0; iter < (1 << 6); iter++){
                sum += arr[start];
                start += (1 << blockSizePower);
            }
        }

        // 2^(numBlocksPower + blockSizePower) to 2^(numBlocksPower + blockSizePower + 5)
        if(n & (1 << (numBlocksPower + blockSizePower + 5))){
            // unrolling 32 times
            #pragma unroll 
            for(unsigned int iter = 0; iter < (1 << 5); iter++){
                sum += arr[start];
                start += (1 << blockSizePower);
            }
        }
        if(n & (1 << (numBlocksPower + blockSizePower + 4))){
            // unrolling 16 times
            #pragma unroll
            for(unsigned int iter = 0; iter < (1 << 4); iter++){
                sum += arr[start];
                start += (1 << blockSizePower);
            }
       }
       if(n & (1 << (numBlocksPower + blockSizePower + 3))){
            // unrolling 8 times 
            #pragma unroll
            for(unsigned int iter = 0; iter < (1 << 3); iter++){
                sum += arr[start];
                start += (1 << blockSizePower);
            }   
       }          
       if(n & (1 << (numBlocksPower + blockSizePower + 2))){
            // unrolling 4 times
            #pragma unroll
            for(unsigned int iter = 0; iter < (1 << 2); iter++){
                sum += arr[start];
                start += (1 << blockSizePower);
            }   
       }   
       if(n & (1 << (numBlocksPower + blockSizePower + 1))){
            // unrolling 2 times
            #pragma unroll
            for(unsigned int iter = 0; iter < (1 << 1); iter++){
                sum += arr[start];
                start += (1 << blockSizePower);
            }   
       }   
       if(n & (1 << (numBlocksPower + blockSizePower + 0))){
            // unrolling 1 time
            #pragma unroll
            for(unsigned int iter = 0; iter < (1 << 0); iter++){
                sum += arr[start];
                start += (1 << blockSizePower);
            }   
      }   
    }

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

        if constexpr (blockSizePower >= 10) sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        if constexpr (blockSizePower >= 9)  sum += __shfl_down_sync(0x0000FFFF, sum, 8); 
        if constexpr (blockSizePower >= 8)  sum += __shfl_down_sync(0x000000FF, sum, 4); 
        if constexpr (blockSizePower >= 7)  sum += __shfl_down_sync(0x0000000F, sum, 2); 
        if constexpr (blockSizePower >= 6)   sum += __shfl_down_sync(0x00000003, sum, 1);

        if (laneId == 0) accum[blockIdx.x] = sum;
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
T computeReduction(T* d_arr, const int n){

    T* accum;

    if(n & ((1 << 30))){
        cudaMalloc(&accum, (512 + 1) * sizeof(T));

        deviceReductionKernel <T, 9, 8> <<< 512 + 1, 256, 32 * sizeof(T) >>> (d_arr, n, accum);
        deviceReductionAccumulate <T, 512> <<< 1, 512, 32 * sizeof(T) >>> (accum);      
    }else{
        cudaMalloc(&accum, (256 + 1) * sizeof(T));

        deviceReductionKernel <T, 8, 8> <<< 256 + 1, 256, 32 * sizeof(T) >>> (d_arr, n, accum);
        deviceReductionAccumulate <T, 256> <<< 1, 256, 32 * sizeof(T) >>> (accum);
    }

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
        generate_random_array <T> (n, -500, 1000, arr);
    }else if(n <= 1e7){
        generate_random_array <T> (n, -10, 100, arr);
    }else if(n <= 1e8){
        generate_random_array <T> (n, -5, 10, arr);
    }else if(n <= 1e9){
        generate_random_array <T> (n, -1, 2, arr);
    }else if(n <= 2*1e9){
        generate_random_array <T> (n, 0, 1, arr);
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

        bool myReductionCorrectness = true;
        double myReductionMedianTime, myReductionMeanTime, myReductionStandardDeviation;
        int myReductionSum;
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
                myReductionSum = sum;

                if(sum != thrustSum){
                    myReductionCorrectness = false;
                }
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

