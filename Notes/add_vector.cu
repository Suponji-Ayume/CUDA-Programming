#include <stdlib.h>
#include <iostream>
using namespace std;

#define N (4096 * 4096)
#define THREADS_PER_BLOCK 512

// CUDA 核函数执行加法操作
__global__ void add(int *d_a, int *d_b, int *d_c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;    // 获取当前线程的索引
    if (tid < N){
        d_c[tid] = d_a[tid] + d_b[tid];
    }
}

int main(void){
    int *a, *b, *c;             // 指向 host 内存上三个向量的指针
    int *d_a, *d_b, *d_c;       // 指向 device 内存上三个向量的指针

    // 查询可用的设备数量
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cout << "Number of CUDA devices: " << deviceCount << endl;

    if (deviceCount == 0) {
        cerr << "No CUDA devices available" << endl;
        return -1;
    }

    // 设置要使用的设备（例如，设备 0）
    int device = 1;
    cudaSetDevice(device);

    // 获取并显示设备属性
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    cout << "Using device " << device << ": " << deviceProp.name << endl;

    long long size = N * sizeof(int); // 需要申请内存空间的大小

    // 为 host 上的三个向量申请空间并分配
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // 为 device 上的三个向量申请空间并分配
    // cudaMalloc 需要改变 d_a, d_b, d_c 三个指针指向的内存空间，即修改三个指针的值，因此需要传入指针的指针，才能在函数内部同步修改指针的值
    // 如果只是传入一级指针，那么这三个指针指向的地址不会改变，只能在函数体内修改他们指向的地址存放的数据
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // 在 host 内存中初始化向量
    for (int i = 0; i < N; i++){
        a[i] = -i;
        b[i] = i * 2;
    }

    // 将 host 上的向量拷贝到 device 上
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

    // 在 device 上开 N 个 block 并行执行 N 个线程，每个线程执行一个加法操作
    add<<<(N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    // 将 device 上的运算结果拷贝到 host 上
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // 打印向量 c
    cout << a[N-1] << "+" << b[N-1] << "=" << c[N-1] << endl;
    

    // 释放 host 和 device 上的内存空间
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}