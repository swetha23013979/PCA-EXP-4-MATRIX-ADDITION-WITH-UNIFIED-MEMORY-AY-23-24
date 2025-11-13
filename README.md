# PCA-EXP-4-MATRIX-ADDITION-WITH-UNIFIED-MEMORY AY 23-24
<h3>NAME:SWETHA D</h3>
<h3>REGISTER NO</212223040222h3>
<h1> <align=center> MATRIX ADDITION WITH UNIFIED MEMORY </h3>
  Refer to the program sumMatrixGPUManaged.cu. Would removing the memsets below affect performance? If you can, check performance with nvprof or nvvp.</h3>

## AIM:
To perform Matrix addition with unified memory and check its performance with nvprof.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
## PROCEDURE:
1.	Setup Device and Properties
Initialize the CUDA device and get device properties.
2.	Set Matrix Size: Define the size of the matrix based on the command-line argument or default value.
Allocate Host Memory
3.	Allocate memory on the host for matrices A, B, hostRef, and gpuRef using cudaMallocManaged.
4.	Initialize Data on Host
5.	Generate random floating-point data for matrices A and B using the initialData function.
6.	Measure the time taken for initialization.
7.	Compute Matrix Sum on Host: Compute the matrix sum on the host using sumMatrixOnHost.
8.	Measure the time taken for matrix addition on the host.
9.	Invoke Kernel
10.	Define grid and block dimensions for the CUDA kernel launch.
11.	Warm-up the kernel with a dummy launch for unified memory page migration.
12.	Measure GPU Execution Time
13.	Launch the CUDA kernel to compute the matrix sum on the GPU.
14.	Measure the execution time on the GPU using cudaDeviceSynchronize and timing functions.
15.	Check for Kernel Errors
16.	Check for any errors that occurred during the kernel launch.
17.	Verify Results
18.	Compare the results obtained from the GPU computation with the results from the host to ensure correctness.
19.	Free Allocated Memory
20.	Free memory allocated on the device using cudaFree.
21.	Reset Device and Exit
22.	Reset the device using cudaDeviceReset and return from the main function.

## PROGRAM:
```
%%cuda
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(float *ip, const int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            C[iy * nx + ix] = A[iy * nx + ix] + B[iy * nx + ix];
        }
    }
}

__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = true;
    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;
            printf("Mismatch at %d: host %f gpu %f\n", i, hostRef[i], gpuRef[i]);
            break;
        }
    }
    printf("\nResult Check: %s\n", match ? "MATCH ✅" : "NO MATCH ❌");
}

int main(int argc, char **argv)
{
    printf("Matrix Addition using CUDA Unified Memory\n");

    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1 << 12;
    int ny = 1 << 12;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: %d x %d\n", nx, ny);

    float *A, *B, *hostRef, *gpuRef;
    CHECK(cudaMallocManaged(&A, nBytes));
    CHECK(cudaMallocManaged(&B, nBytes));
    CHECK(cudaMallocManaged(&hostRef, nBytes));
    CHECK(cudaMallocManaged(&gpuRef, nBytes));

    double start = seconds();
    initialData(A, nxy);
    initialData(B, nxy);
    double initTime = seconds() - start;
    printf("Initialization Time: %f sec\n", initTime);

    // CPU execution
    start = seconds();
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    double cpuTime = seconds() - start;
    printf("Sum Matrix of CPU Time: %f sec\n", cpuTime);

    dim3 block(32, 32);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    // Warm-up
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1);
    CHECK(cudaDeviceSynchronize());

    // GPU execution
    start = seconds();
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double gpuTime = seconds() - start;
    printf("Sum Matrix of GPU Time: %f sec <<<grid(%d,%d), block(%d,%d)>>>\n",
           gpuTime, grid.x, grid.y, block.x, block.y);

    checkResult(hostRef, gpuRef, nxy);

    CHECK(cudaFree(A));
    CHECK(cudaFree(B));
    CHECK(cudaFree(hostRef));
    CHECK(cudaFree(gpuRef));
    CHECK(cudaDeviceReset());

    return 0;
}
```
## OUTPUT:
<img width="555" height="130" alt="508721573-d4526b94-d02e-45c4-afc9-504c85020fde" src="https://github.com/user-attachments/assets/df144c96-8df6-4869-8b67-917247dc5bec" />

## RESULT:
Thus the program has been executed by using unified memory. It is observed that removing memset function has given less/more 0.008852 time.
