#include <iostream>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include <fstream>

using namespace std;

__global__ void mandlebrot(float *v, float minX, float maxX, float minY, float maxY, int width, int height, int N, int msX, int msY)
{
    double exponent = 1.3;
    double multiplier = 1;

    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    float baseX, baseY, fx, fy, a, b;

    baseX = minX + (maxX - minX) * ((float)xIndex / (float)width);
    baseY = minY + (maxY - minY) * ((float)yIndex / (float)height);

    float xInt = (maxX - minX) * 1.0 / float(width);
    float yInt = (maxY - minY) * 1.0 / float(height);

    double count;
    double val = 0;
    for (int i = 0; i < msX; i++)
    {
        float x = baseX - (msX - 1) / (2*msX) + xInt * i/(msX);
        for (int j = 0; j < msY; j++)
        {
            float y = baseY - (msY - 1) / (2*msY) + yInt * i/(msY);
            a = 0;
            b = 0;
            for (int k = 0; k < N; k++)
            {

                fx = a * a - b * b + x;
                fy = 2 * a * b + y;
                a = fx;
                b = fy;

                count = k;
                if (fx * fx + fy * fy >= 4)
                {
                    break;
                }
            }
            double mag = fx * fx + fy * fy;
            if (mag > 4)
            {
                val += fmod(pow(exponent,count) * multiplier, 256.0);
            }
            else
            {
                val += 0;
            }
        }
    }
    val /= msX * msY;
    if (val > 0)
    {
        if (val < 256)
        {
            v[xIndex + yIndex * width] = 0;
            v[xIndex + yIndex * width + width * height] = val;
            v[xIndex + yIndex * width + width * height * 2] = 0;
        }
        else if (val < 512)
        {
            v[xIndex + yIndex * width] = 255;
            v[xIndex + yIndex * width + width * height] = (int)fmod(val,256.0);
            v[xIndex + yIndex * width + width * height * 2] = 0;
        }
        else if (val < 768)
        {
            v[xIndex + yIndex * width] = 256 - (int)fmod(val, 256.0);
            v[xIndex + yIndex * width + width * height] = 255;
            v[xIndex + yIndex * width + width * height * 2] = 0;
        }
        else if (val < 1024)
        {
            v[xIndex + yIndex * width] = 0;
            v[xIndex + yIndex * width + width * height] = 255;
            v[xIndex + yIndex * width + width * height * 2] = (int)fmod(val, 256.0);
        }
        else if (val < 1280)
        {
            v[xIndex + yIndex * width] = 0;
            v[xIndex + yIndex * width + width * height] = 256 - (int)fmod(val, 256.0);
            v[xIndex + yIndex * width + width * height * 2] = 255;
        }
        else if (val < 1536)
        {
            v[xIndex + yIndex * width] = (int)fmod(val, 256.0);
            v[xIndex + yIndex * width + width * height] = 0;
            v[xIndex + yIndex * width + width * height * 2] = 255;
        }
        else if (val < 1792)
        {
            v[xIndex + yIndex * width] = 255;
            v[xIndex + yIndex * width + width * height] = 0;
            v[xIndex + yIndex * width + width * height * 2] = 256 - (int)fmod(val, 256.0);
        }
    }
    else
    {
        v[xIndex + yIndex * width] = 0;
        v[xIndex + yIndex * width + width * height] = 0;
        v[xIndex + yIndex * width + width * height * 2] = 0;
    }
}

void save_values(float *v, int width, int height)
{
    ofstream img("picture.ppm");
    img << "P3" << endl;
    img << width << " " << height << endl;
    img << "255" << endl;
    for (int i = 0; i < width * height; i++)
    {
        img << int(v[i]) << ' ' << int(v[i + width * height]) << ' ' << int(v[i + width * height * 2]) << endl;
    }
}

void print_values(float *v, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (i % 1080 == 0)
        {
            cout << endl;
        }
        cout << v[i] << ',' << v[i + N] << ',' << v[i + N * 2] << ' ';
    }
}
int main()
{
    int N = 500;
    int width = 1920;
    int height = 1080;
    int msX = 16;
    int msY = 16;
    float minX = -2.5;
    float maxX = 1.5;
    float minY = -18 / 16;
    float maxY = 18 / 16;
    size_t bytes = width * height * sizeof(float) * 3;

    float *a;
    cudaMallocManaged(&a, bytes);

    int threadsX = 32;
    int threadsY = 18;

    int blocks = width / threadsX;

    dim3 THREADS(threadsX, threadsY);
    dim3 BLOCKS(blocks, blocks);

    mandlebrot<<<BLOCKS, THREADS>>>(a, minX, maxX, minY, maxY, width, height, N, msX, msY);
    cudaDeviceSynchronize();

    save_values(a, width, height);
    return 1;
}