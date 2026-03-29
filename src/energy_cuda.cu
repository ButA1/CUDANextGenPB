#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include "test.h"

void init(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    // std::random_device dev;
    std::mt19937 prng(2024);
    std::uniform_int_distribution<int32_t> distrib(-16, 16);

    for (auto i = 0; i < size; i++)
    {
        vec_a[i] = distrib(prng);
        vec_b[i] = distrib(prng);
    }

    for (auto i = 0; i < size * size; i++)
        mat[i] = distrib(prng);
}

__global__ void computeVector(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *tmp){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        tmp[i] = vec_a[i] + vec_b[i];
    }
}

__global__ void computeMatrix(int32_t size, int32_t *mat, int32_t *tmp, int32_t *out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t sum = 0;
    if(i < size){
        for(int j = 0; j < size; j++){
            sum += tmp[j] * mat[i * size + j];
        }
        out[i] = sum;
    }
}

void check(cudaError_t err, std::string msg) {
  if (err != cudaSuccess) {
    std::cerr << (cudaGetErrorString(err)) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void pretty_print(int32_t size, int32_t *vec_a, int32_t *vec_b, int32_t *mat)
{
    std::cout << "Vec A:" << std::endl;
    for (auto i = 0; i < size; i++)
        std::cout << vec_a[i] << std::endl;

    std::cout << "Vec B:" << std::endl;
    for (auto i = 0; i < size; i++)
        std::cout << vec_b[i] << std::endl;

    std::cout << "Matrix:" << std::endl;
    for (auto i = 0; i < size; i++)
    {
        for (auto j = 0; j < size; j++)
            std::cout << mat[i * size + j] << " ";

        std::cout << std::endl;
    }
}

// TODO
int energy_cuda(ray_cache_t & ray_cache)
{
    // int32_t size = 3;
    int32_t size = 32768;

    auto vec_a = (int32_t *)malloc(sizeof(int32_t) * size);
    auto vec_b = (int32_t *)malloc(sizeof(int32_t) * size);

    auto tmp = (int32_t *)malloc(sizeof(int32_t) * size);
    // Flat Buffer for matrix
    auto mat = (int32_t *)malloc(sizeof(int32_t *) * size * size);
    auto out_gpu = (int32_t *)malloc(sizeof(int32_t) * size);

    for (int i = 0; i < size; i++){
        out_gpu[i] = 0;
    }

    // malloc cuda memory
    cudaError_t err = cudaSuccess;
    int32_t * vec_a_cu = NULL; 
    int32_t * vec_b_cu = NULL;
    int32_t * tmp_cu = NULL;
    int32_t * mat_cu = NULL;
    int32_t * out_cu = NULL;
    err = cudaMalloc((void **)&vec_a_cu,sizeof(int32_t) * size);
    check(err, "Failed to allocate device memory for vec_a");
    err = cudaMalloc((void **)&vec_b_cu,sizeof(int32_t) * size);
    check(err, "Failed to allocate device memory for vec_b");
    err = cudaMalloc((void **)&tmp_cu,sizeof(int32_t) * size);
    check(err, "Failed to allocate device memory for tmp");
    err = cudaMalloc((void **)&mat_cu,sizeof(int32_t) * size * size);
    check(err, "Failed to allocate device memory for mat");
    err = cudaMalloc((void **)&out_cu,sizeof(int32_t) * size);
    check(err, "Failed to allocate device memory for out");

    init(size, vec_a, vec_b, mat);

    // pretty_print(size, vec_a, vec_b, mat);

    auto start = std::chrono::system_clock::now();
    // copy data to device
    err = cudaMemcpy(vec_a_cu, vec_a, sizeof(int32_t) * size, cudaMemcpyHostToDevice);
    check(err, "Failed to copy vec_a from host to device");
    err = cudaMemcpy(vec_b_cu, vec_b, sizeof(int32_t) * size, cudaMemcpyHostToDevice);
    check(err, "Failed to copy vec_b from host to device");
    err = cudaMemcpy(mat_cu, mat, sizeof(int32_t) * size * size, cudaMemcpyHostToDevice);
    check(err, "Failed to copy mat from host to device");

    // add kernel call
    int threadsPerBlock = 1024;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    computeVector<<<blocksPerGrid, threadsPerBlock>>>(size, vec_a_cu, vec_b_cu, tmp_cu);
    err = cudaGetLastError();
    check(err, "Failed to launch computeVector kernel");
    cudaDeviceSynchronize();

    computeMatrix<<<blocksPerGrid, threadsPerBlock>>>(size, mat_cu, tmp_cu, out_cu);
    err = cudaGetLastError();
    check(err, "Failed to launch computeMatrix kernel");
    cudaDeviceSynchronize();

    // copy data back to host
    err = cudaMemcpy(out_gpu, out_cu, sizeof(int32_t) * size, cudaMemcpyDeviceToHost);
    check(err, "Failed to copy out from device to host");
    auto end = std::chrono::system_clock::now();

    // time taken for GPU
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time GPU: " << elapsed_seconds.count() << "s" << std::endl;

    // time taken for CPU
    elapsed_seconds = end - start;
    std::cout << "Elapsed time CPU: " << elapsed_seconds.count() << "s" << std::endl;

    free(vec_a);
    free(vec_b);
    free(mat);

    err = cudaFree(vec_a_cu);
    check(err, "Failed to free device memory for vec_a");
    err = cudaFree(vec_b_cu);
    check(err, "Failed to free device memory for vec_b");
    err = cudaFree(tmp_cu);
    check(err, "Failed to free device memory for tmp");
    err = cudaFree(mat_cu);
    check(err, "Failed to free device memory for mat");
    err = cudaFree(out_cu);
    check(err, "Failed to free device memory for out");

    return 0;
}
