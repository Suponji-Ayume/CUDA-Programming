#include <stdlib.h>
#include <iostream>
using namespace std;

# define BLOCK_SIZE 7
# define RADIUS 3

// 一维 block 模板
__global__ void stencil_1D(int *in, int *out){
    // 创建属于当前 block 的共享内存
    // 主体部分大小需要和 block 线程块中的线程数一致，这样能保证每一个线程的计算得到的临时变量都有唯一的存储位置
    // 主体两侧都设置有缓冲区，用来存放需要协同当前 block 处理的，但是被划分到前后两个 block 主体中的数据
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程的全局索引
    int local_idx = threadIdx.x + RADIUS;                   // 计算当前线程与当前块的共享内存对应的局部索引，该线程计算得到的结果就存放在共享内存的该索引处

    // 将全局内存中的数据拷贝到共享内存中
    temp[local_idx] = in[global_idx];

    // 处理边界情况，将前后两个 block 中的数据拷贝到当前 block 的共享内存中
    if (threadIdx.x < RADIUS){
        temp[local_idx - RADIUS] = in[global_idx - RADIUS];
        temp[local_idx + BLOCK_SIZE] = in[global_idx + BLOCK_SIZE];
    }

    // 在进行具体计算之前，要求当前 block 中的所有线程都完成数据的拷贝
    // 因此需要设置一个 barrier, 保证同步线程
    __syncthreads();
    
    // 这里模拟一维卷积操作
    // 假设卷积核大小为 7，那么 RADIUS 的大小就是 (7 - 1) / 2 = 3
    int thread_result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++){
        thread_result += temp[local_idx + offset];
    }

    // 由于这里每个线程计算完成后，没有后续用到这些结果的计算，也就不需要再将临时结果复写入共享内存中
    // 直接将计算结果写入全局内存中
    out[global_idx] = thread_result;

}
int main(){
    return 0;
}