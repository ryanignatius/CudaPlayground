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
    std::cout << "Reduce 0: " << d_out[0] << std::endl;

    if (numBlocks > 1){
        ret += reduce0(numBlocks, numThread, d_out);
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

    int red0 = reduce0(N, blockSize, d_in);
    std::cout << "Reduce 0: " << red0 << std::endl;
    
    cudaFree(d_in);

    return 0;
}
