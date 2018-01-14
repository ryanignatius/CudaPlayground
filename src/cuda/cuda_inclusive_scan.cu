#include <iostream>
#include <math.h>
#include <ctime>
#include <algorithm>

__global__ void scan(int n, int *d_in, int *d_out, int *d_temp){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int id = threadIdx.x;
    int m = min(n, blockDim.x);

    extern __shared__ int sdata[];
    sdata[id] = d_in[index];
    __syncthreads();

    int step = 1;
    while (step < m){
        int cur = sdata[id];
        int ileft = id - step;
        if (ileft >= 0){
            cur += sdata[ileft];
        }
        __syncthreads();
        sdata[id] = cur;
        __syncthreads();
        step *= 2;
    }

    d_out[index] = sdata[id];
    if (id == m-1){
        d_temp[blockIdx.x] = sdata[id];
    }
}

__global__ void add(int n, int *d_in, int *d_out, int *d_temp){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int id = blockIdx.x;

    if (id == 0){
        d_out[index] = d_in[index];
    } else {
        d_out[index] = d_in[index] + d_temp[id-1];
    }
}

void scan(int n, int *h_in, int *h_out){
    int blockSize = min(n, 512);
    int numBlocks = (n + blockSize - 1) / blockSize;
    int *temp;
    cudaMallocManaged(&temp, numBlocks*sizeof(int));

    scan<<<numBlocks, blockSize, blockSize*sizeof(int)>>>(n, h_in, h_out, temp);
    cudaDeviceSynchronize();
    if (numBlocks > 1){
        scan(numBlocks, temp, temp);
    }
    add<<<numBlocks, blockSize>>>(n, h_out, h_out, temp);
    cudaDeviceSynchronize();

    cudaFree(temp);
}

int main(){
    std::clock_t startGPU, endGPU;
    std::clock_t startCPU, endCPU;
    int smallN = 1<<6;
    int largeN = 1<<26;

    // GPU version
    int *x;
    cudaMallocManaged(&x, smallN*sizeof(int));
    int *xout;
    cudaMallocManaged(&xout, smallN*sizeof(int));

    int *y;
    cudaMallocManaged(&y, largeN*sizeof(int));
    int *yout;
    cudaMallocManaged(&yout, largeN*sizeof(int));

    for (int i=0; i<smallN; i++){
        x[i] = (i+1);
    }
    for (int i=0; i<largeN; i++){
        y[i] = 10;
    }

    startGPU = std::clock();
    scan(smallN, x, xout);
    scan(largeN, y, yout);
    endGPU = std::clock();

    // CPU version
    startCPU = std::clock();
    int *smallArr = new int[1<<6];
    int *largeArr = new int[1<<26];
    for (int i=0; i<smallN; i++){
        smallArr[i] = (i+1);
        if (i > 0) smallArr[i] += smallArr[i-1];
    }
    for (int i=0; i<largeN; i++){
        largeArr[i] = 10;
        if (i > 0) largeArr[i] += largeArr[i-1];
    }
    endCPU = std::clock();

    // compare result:
    for (int i=0; i<smallN; i++){
        std::cout << x[i] << ": " << xout[i] << " " << smallArr[i] << std::endl;
    }
    for (int i=1; i<largeN; i*=10){
        std::cout << i << ": " << yout[i] << " " << largeArr[i] << std::endl;
    }
    std::cout << "Scan: " << yout[largeN-1] << " " << largeArr[largeN-1] << std::endl;

    // time
    std::cout << "GPU Time: " << ((endGPU - startGPU) / (double) CLOCKS_PER_SEC) << std::endl;
    std::cout << "CPU Time: " << ((endCPU - startCPU) / (double) CLOCKS_PER_SEC) << std::endl;

    cudaFree(x);
    cudaFree(xout);
    cudaFree(y);
    cudaFree(yout);
    return 0;
}
