#include <iostream>
#include <cassert>

// Initalises a sample vector with values [0,max)
void initVector(float *v, int N, int max){
    for(int i=0; i<N; i++){
        v[i] = (rand()) % 32768 / 32768.0 * (max);      
    }
}

void print(float* vector, int N){
    for (int i=0; i<N; i++){
        std::cout << vector[i] << ' ';
    }
    std::cout << '\n';
}

void print(float* matrix, int N, int M){
    for (int i=0; i<M; i++){
        for (int j=0; j<N; j++){
            std::cout << matrix[j + i * N] << ' ';
        }
        std::cout << '\n';
    }
}

void print(float* matrix, int N, int M, int printRow){
    for (int i=0; i<M; i++){
        for (int j=0; j<printRow; j++){
            std::cout << matrix[j + i * N] << ' ';
        }
        std::cout << '\n';
    }
}

// Adds a vector elementwise to another inplace A (also works with tensors of any order)
__global__ void vectorAdd(float* a, float* b, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N){
        a[i] = a[i] + b[i];
    }
}

// Subtracts a vector elementwise from another inplace A (also works with tensors of any order)
__global__ void vectorSubtract(float* a, float* b, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N){
        a[i] = a[i] - b[i];
    }
}

// Scales a vector by a floating point (a vector of one element) (also works with tensors of any order) 
__global__ void vectorScale(float* a, float* b, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N){
        a[i] = a[i]*b[0];
    }
}

// Multiplies a vector element wise (also works with tensors of any order) and stores the output inplace a
__global__ void vectorMult(float* a, float* b, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N){
        a[i] = a[i]*b[i];
    }
}

// Uses each combination of the elements of the two vectors to create a corresponding matrix
__global__ void outerProduct(float* a, float* b, float* c, int N, int M){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < M){
        c[i*N + j] = a[i] * b[j];
    }
}

// Sums the elementwise multiples of each row of the matrix and the vector into the corresponding element of the new vector
__global__ void matrixVectorMult(float* a, float* b, float* c, int N, int M){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    c[j] = 0;
    for (int i=0; i<M; i++){
        c[j] += a[i + j*N] * b[i];
    }
}

void vectorAdd(float* a, float* b, float* c, int N, int threadSize){
    float *da, *db;
    size_t bytes = N * sizeof(float);

    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

    int threads = threadSize;
    int blocks = (N +threads-1)/threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    vectorAdd<<<BLOCKS,THREADS>>>(da,db,N);
    cudaDeviceSynchronize();

    cudaMemcpy(c, da, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(da);
    cudaFree(db);
}

void vectorAdd(float* a, float* b, int N, int threadSize){
    float *da, *db;
    size_t bytes = N * sizeof(float);

    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

    int threads = threadSize;
    int blocks = (N +threads-1)/threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    vectorAdd<<<BLOCKS,THREADS>>>(da,db,N);
    cudaDeviceSynchronize();

    cudaMemcpy(a, da, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(da);
    cudaFree(db);
}

void vectorSubtract(float* a, float* b, float* c, int N, int threadSize){
    float *da, *db;
    size_t bytes = N * sizeof(float);

    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

    int threads = threadSize;
    int blocks = (N +threads-1)/threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    vectorSubtract<<<BLOCKS,THREADS>>>(da,db,N);
    cudaDeviceSynchronize();

    cudaMemcpy(c, da, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(da);
    cudaFree(db);
}

void vectorSubtract(float* a, float* b, int N, int threadSize){
    float *da, *db;
    size_t bytes = N * sizeof(float);

    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

    int threads = threadSize;
    int blocks = (N +threads-1)/threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    vectorSubtract<<<BLOCKS,THREADS>>>(da,db,N);
    cudaDeviceSynchronize();

    cudaMemcpy(a, da, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(da);
    cudaFree(db);
}

void vectorScale(float* a, float* b, int N, int threadSize){
    float *da, *db;
    size_t bytes = N * sizeof(float);

    cudaMalloc(&da, bytes);
    cudaMalloc(&db, sizeof(float));
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sizeof(float), cudaMemcpyHostToDevice);

    int threads = threadSize;
    int blocks = (N +threads-1)/threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    vectorScale<<<BLOCKS,THREADS>>>(da,db,N);
    cudaDeviceSynchronize();

    cudaMemcpy(a, da, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(da);
    cudaFree(db);
}

void vectorMult(float* a, float* b, float* c, int N, int threadSize){
    float *da, *db;
    size_t bytes = N * sizeof(float);

    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

    int threads = threadSize;
    int blocks = (N +threads-1)/threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    vectorMult<<<BLOCKS,THREADS>>>(da,db,N);
    cudaDeviceSynchronize();

    cudaMemcpy(c, da, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(da);
    cudaFree(db);
}

void vectorMult(float* a, float* b, int N, int threadSize){
    float *da, *db;
    size_t bytes = N * sizeof(float);

    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);

    int threads = threadSize;
    int blocks = (N +threads-1)/threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    vectorMult<<<BLOCKS,THREADS>>>(da,db,N);
    cudaDeviceSynchronize();

    cudaMemcpy(a, da, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(da);
    cudaFree(db);
}

void outerProduct(float* a, float* b, float* c, int N, int M, int threadSize){
    float *da, *db, *dc;
    size_t bytesA = N * sizeof(float);
    size_t bytesB = M * sizeof(float);
    size_t bytesC = N*M * sizeof(float);

    cudaMalloc(&da, bytesA);
    cudaMalloc(&db, bytesB);
    cudaMalloc(&dc, bytesC);
    cudaMemcpy(da, a, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytesB, cudaMemcpyHostToDevice);

    int threads = threadSize;
    int blocksX = (N +threads-1)/threads;
    int blocksY = (M +threads-1)/threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocksX, blocksY);

    outerProduct<<<BLOCKS,THREADS>>>(da,db,dc,N,M);
    cudaDeviceSynchronize();

    cudaMemcpy(c, dc, bytesC, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

void matrixVectorMult(float* a, float* b, float* c, int N, int M, int threadSize){
    float *da, *db, *dc;
    size_t bytesA = N*M * sizeof(float);
    size_t bytesB = M * sizeof(float);
    size_t bytesC = N * sizeof(float);

    cudaMalloc(&da, bytesA);
    cudaMalloc(&db, bytesB);
    cudaMalloc(&dc, bytesC);
    cudaMemcpy(da, a, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, bytesB, cudaMemcpyHostToDevice);

    int threads = threadSize;
    int blocks = (M +threads-1)/threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    matrixVectorMult<<<BLOCKS,THREADS>>>(da,db,dc,N,M);
    cudaDeviceSynchronize();

    cudaMemcpy(c, dc, bytesC, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

int main(){
    std::clock_t tStart = clock();

    int N = 50;

    float *a, *b;

    a = (float*) malloc(sizeof(float) * N);
    b = (float*) malloc(sizeof(float) * N);
    float k = 100;

    initVector(a,N,2);
    initVector(b,N,2);
    print(a,10);
    print(a,0);
    print(b,10);
    print(a,0);

    vectorScale(b, &k, N, 32);
    vectorAdd(a,b,N,32);
    print(a, 10);

    free(a);
    free(b);

    double timeElapsed = (double)(clock() - tStart) / CLOCKS_PER_SEC;
    std::cout << "CPU Time: " << timeElapsed << '\n';

    return 0;
}