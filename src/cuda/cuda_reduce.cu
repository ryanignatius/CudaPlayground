#include <iostream>
#include <math.h>

__global__ void reduce0(int *d_in, int *d_out){
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();
    
    for (unsigned int s=1; s<blockDim.x; s*=2){
        if (tid % (2*s) == 0){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    
    if (tid == 0){
        d_out[blockIdx.x] = sdata[0];
    }
}

int reduce0(int n, int numThread, int *d_in){
    int ret = 0;

    int *d_out;
    cudaMallocManaged(&d_out, n*sizeof(int));

    int blockSize = numThread;
    if (n < blockSize){
        blockSize = n;
    }
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    reduce0<<<numBlocks, blockSize, blockSize*sizeof(int)>>>(d_in, d_out);
    cudaDeviceSynchronize();
//    std::cout << "Reduce 0: " << d_out[0] << std::endl;

    if (numBlocks > 1){
        ret += reduce0(numBlocks, numThread, d_out);
    } else {
        ret += d_out[0];
    }

    cudaFree(d_out);
    return ret;
}

__global__ void reduce1(int *d_in, int *d_out){
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();
    
    for (unsigned int s=1; s<blockDim.x; s*=2){
        int index = 2 * s * tid;
        if (index < blockDim.x){
            sdata[index] += sdata[index+s];
        }
        __syncthreads();
    }
    
    if (tid == 0){
        d_out[blockIdx.x] = sdata[0];
    }
}

int reduce1(int n, int numThread, int *d_in){
    int ret = 0;

    int *d_out;
    cudaMallocManaged(&d_out, n*sizeof(int));

    int blockSize = numThread;
    if (n < blockSize){
        blockSize = n;
    }
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    reduce1<<<numBlocks, blockSize, blockSize*sizeof(int)>>>(d_in, d_out);
    cudaDeviceSynchronize();
//    std::cout << "Reduce 0: " << d_out[0] << std::endl;

    if (numBlocks > 1){
        ret += reduce1(numBlocks, numThread, d_out);
    } else {
        ret += d_out[0];
    }

    cudaFree(d_out);
    return ret;
}

__global__ void reduce2(int *d_in, int *d_out){
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = d_in[i];
    __syncthreads();
    
    for (unsigned int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    
    if (tid == 0){
        d_out[blockIdx.x] = sdata[0];
    }
}

int reduce2(int n, int numThread, int *d_in){
    int ret = 0;

    int *d_out;
    cudaMallocManaged(&d_out, n*sizeof(int));

    int blockSize = numThread;
    if (n < blockSize){
        blockSize = n;
    }
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    reduce2<<<numBlocks, blockSize, blockSize*sizeof(int)>>>(d_in, d_out);
    cudaDeviceSynchronize();
//    std::cout << "Reduce 0: " << d_out[0] << std::endl;

    if (numBlocks > 1){
        ret += reduce2(numBlocks, numThread, d_out);
    } else {
        ret += d_out[0];
    }

    cudaFree(d_out);
    return ret;
}

__global__ void reduce3(int *d_in, int *d_out){
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    sdata[tid] = d_in[i] + d_in[i+blockDim.x];
    __syncthreads();
    
    for (unsigned int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s){
            sdata[tid] += sdata[tid+s];
        }
        __syncthreads();
    }
    
    if (tid == 0){
        d_out[blockIdx.x] = sdata[0];
    }
}

int reduce3(int n, int numThread, int *d_in){
    int ret = 0;

    int *d_out;
    cudaMallocManaged(&d_out, n*sizeof(int));

    int blockSize = numThread;
    if (n < blockSize){
        blockSize = n;
    }
    int numBlocks = (n + blockSize - 1) / blockSize;
    if (numBlocks > 1){
        reduce3<<<numBlocks/2, blockSize, blockSize*sizeof(int)>>>(d_in, d_out);
    } else {
        reduce3<<<numBlocks, blockSize/2, (blockSize/2)*sizeof(int)>>>(d_in, d_out);
    }
    cudaDeviceSynchronize();
//    std::cout << "Reduce 0: " << d_out[0] << std::endl;

    if (numBlocks > 2){
        ret += reduce3(numBlocks, numThread, d_out);
    } else {
        ret += d_out[0];
    }

    cudaFree(d_out);
    return ret;
}

int main(){
    int N = 1<<26;
    int blockSize = 512;
    
    int *d_in;
    cudaMallocManaged(&d_in, N*sizeof(int));
    for (int i=0; i<N; i++){
        d_in[i] = 10;
    }

    std::cout << "Reduce 0: " << reduce0(N, blockSize, d_in) << std::endl;
    std::cout << "Reduce 1: " << reduce1(N, blockSize, d_in) << std::endl;
    std::cout << "Reduce 2: " << reduce2(N, blockSize, d_in) << std::endl;
    std::cout << "Reduce 3: " << reduce3(N, blockSize, d_in) << std::endl;
    
    /*
     GPU activities:   46.31%  22.244ms         3  7.4145ms  4.5440us  22.190ms  reduce0(int*, int*)
                   24.47%  11.752ms         3  3.9173ms  3.4880us  11.721ms  reduce1(int*, int*)
                   19.71%  9.4677ms         3  3.1559ms  3.2640us  9.4417ms  reduce2(int*, int*)
                    9.51%  4.5690ms         3  1.5230ms  3.2000us  4.5509ms  reduce3(int*, int*)
    */

    cudaFree(d_in);

    return 0;
}
