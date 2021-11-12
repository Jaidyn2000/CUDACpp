#include <iostream>
#include <cstdlib>
#include <time.h>

using namespace std;

void init_vector(float *v, int N){
    for(int i=0; i<N; i++){
        v[i] = ((float)rand()) /((float)32768);      
    }
}

__global__ void vectorAdd(float *a,float *b,float *c,int N){
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x<N){
        c[x] = a[x] + b[x];
    }
}

__global__ void vectorMultDot(float *a, float *b, float *c, int N){
  
    for (int i=0; i<N; i++){
        c[0] += a[i] * b[i];
    }
}
__global__ void vectortoRank2(float *a, float *b, float *c, int N){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    c[x + N * y] = a[x] * b[y];
}

void printValue(float *v, int i){
    cout<< v[i]<< ' ';
}

void printValues(float *v,int N, int O, int add){
    for (int i=0; i<N; i++){
        if (O == 1){
            printValue(v,i + add);
        }else if (O==0){
            printValue(v,0);
            i +=N;
        }
        else{
            printValues(v,i,O-1,add);
                cout<< endl;
                add += N ^ (O-1);
        }
    }
}

int main(){
    int N = 1024;
    int O = 2;
    size_t bytes = N * sizeof(float);
    size_t bytesOut = N ^ O * sizeof(float);

    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytesOut);

    init_vector(a,N);
    init_vector(b,N);
    
    int threads = 16;
    int blocks = (N +threads-1)/threads;

    dim3 THREADS(threads,threads);
    dim3 BLOCKS(blocks,blocks);

    //vectorAdd<<<BLOCKS,THREADS>>>(a,b,c,N);
    //vectorMultDot<<<1,1>>>(a,b,c,N);
    vectortoRank2<<<BLOCKS,THREADS>>>(a,b,c,N);
    cudaDeviceSynchronize();

    printValues(c,N,O,0);
    cout << "Program completed successfully!";
    cin.get();
    return 1;
}