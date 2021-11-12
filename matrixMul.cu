#include <iostream>
#include <cassert>

using namespace std;

__global__ void matrixMul(float *a, float *b, float *c, int N){
    // Calculate row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column =  blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check for matrix
    if(row < N && column < N){
        float temp = 0;
        for (int i=0; i<N; i++){
            temp += a[(i,row)] * b[(column,i)];
        }

        // Write back the result
        c[(column,row)] = temp;

    }
}

void init_matrix(float *m, int N){
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            m[(i,j)] = rand() %1;
        }
    }
}

// Verify the result on the CPU
void verifyResult(float *a, float *b, float *c, int N){
    float temp;
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            temp = 0;
            for (int k=0; k<N; k++){
                temp += a[(k,i)] * b[(j,k)];
            }
            // Check each result
            cout << c[(i,j)]<< ' ';
        }
        cout << endl;
    }
}
int main(){
    // Set our matrix dimensions
    int N = 1024;
    size_t bytes = N * N * sizeof(float);
    cout << bytes;
    // Allocate memory for our matrices
    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize data
    init_matrix(a, N);
    init_matrix(b, N);

    // Set our block and grid dimensions
    int threads = 16;
    int blocks = (N+threads -1)/threads;

    // Setup our kernel launch parameters
    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks,blocks);

    // Launch our kernel
    matrixMul<<<BLOCKS,THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize();

    // Verify the result
    verifyResult(a,b,c,N);

    cout << "Program completed successfully!";
    return 0;
}