#include<iostream>
#include<string>
#include<cstring>
#include "EasyBMP.hpp"
// #include "mpi.h"
// #include "cuda.h"
// #include "cuda_runtime_api.h"
using namespace std;
using namespace EasyBMP;
#define ROW_PER_THREAD 16 
#define COL_PER_THREAD 16 
__global__ void  heatConductionKernel(double* new_data, double* data,int width,int height, double K) {
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = 0;i < ROW_PER_THREAD;i++) {
        for (int j = 0;j < COL_PER_THREAD;j++){
            int x = grid_x * ROW_PER_THREAD + i;
            int y = grid_y * COL_PER_THREAD + j;
            if(x<width && y <height){
                double delta = (K) * (data [(x - 1) * width + y] + data [(x + 1) * width + y] + data [x * width + y - 1] + data [x * width + y + 1] - 4 * data [x * width + y]);
                new_data[x * width + y] = data[x * width + y] + delta;
            }
        }
    }
}
class HeatConduction {
public:
    double K;
    int height;
    int width;
    double* data;
    double* check_data;
    double range_max = 1000;
    HeatConduction(double K, double height, double width) {
        this->K = K;
        this->height = height;
        this->width = width;
        data =  (double*) malloc(sizeof(double) * width * height);
    }
    void loadData(double* data) {
        for (int i = 0; i < height * width; i++) {
            this->data [i] = data [i];
        }
    }
    void loadDataFromFile(char* fileName) {
        FILE* file = fopen(fileName, "r");
        for (int i = 0; i < height * width; i++) {
            fscanf(file, "%lf", &data [i]);
        }
        fclose(file);
    }
    void saveDataToFile(char* fileName) {
        FILE* file = fopen(fileName, "w");
        for (int i = 0; i < height * width; i++) {
            fprintf(file, "%lf\n", data [i]);
        }
        fclose(file);
    }
    void printData() {
        for (int i = 0; i < height * width; i++) {
            cout << data [i] << " ";
        }
        cout << endl;
    }
    double find_max() {
        double ret = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (data [i * width + j] > ret) {
                    ret = data [i * width + j];
                }
            }
        }
        return ret;
    }
    void save2img(string filename) {
        double range_max = max(find_max(), this->range_max);
        if (height <= 1000 && width <= 1000) {
            EasyBMP::RGBColor black(0, 0, 0);
            Image img(width, height, filename, black);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    RGBColor color(data [i * width + j] / range_max, data [i * width + j] / range_max, data [i * width + j] / range_max);
                    color.r = 255 - color.r;
                    color.g = 255 - color.g;
                    color.b = 255 - color.b;
                    img.SetPixel(j, i, color);
                }
            }
            img.Write();
        }
    }
    //cpu单线程
    void cpuSolver(int iteration) {
        double* new_data = (double*)malloc(sizeof(double) * height * width);
        for (int k = 0;k < iteration;k++) {
            for (int i = 0;i < height;i++) {
                for (int j = 0;j < width;j++) {
                    //边界条件
                    if (i == 0 || i == height - 1 || j == 0 || j == width - 1) {
                        new_data [i * width + j] = data [i * width + j];
                    }
                    else {
                        double delta = (K) * (data [(i - 1) * width + j] + data [(i + 1) * width + j] + data [i * width + j - 1] + data [i * width + j + 1] - 4 * data [i * width + j]);
                        new_data [i * width + j] = data [i * width + j] + delta;
                    }
                }
            }
            //更新新的数据
            for (int i = 0;i < height;i++) {
                for (int j = 0;j < width;j++) {
                    data [i * width + j] = new_data [i * width + j];
                }
            }
        }
        memcpy(check_data, data, sizeof(double) * height * width);
        free(new_data);
    }
    //单gpu
    /*
    block size: 16*16
    each thread : 16*16
    */
    void gpuSolver(int iteration) {
        dim3 dimBlock(16, 16);
        dim3 dimGrid((int)width / ROW_PER_THREAD / dimBlock.x + 1, (int)height / COL_PER_THREAD / dimBlock.y);
        double* d_data, * d_new_data;
        cudaMalloc(&d_data, sizeof(double) * (int)height * (int)width);
        cudaMalloc(&d_new_data, sizeof(double) * (int)height * (int)width);
        cudaMemcpy(d_data, data, sizeof(double) * (int)height * (int)width, cudaMemcpyHostToDevice);
        for (int k = 0;k < iteration;k++) {
            heatConductionKernel << <dimGrid, dimBlock >> > (d_new_data,d_data,width, height, K);
            cudaMemcpy(d_data, d_new_data, sizeof(double) * (int)height * (int)width, cudaMemcpyDeviceToDevice);
        }
        cudaMemcpy(data, d_data, sizeof(double) * (int)height * (int)width, cudaMemcpyDeviceToHost);
        cudaFree(d_data);
        cudaFree(d_new_data);
    }
    void check(){
        double max_diff = 0;
        for (int i = 0;i < height;i++) {
            for (int j = 0;j < width;j++) {
                if (abs(data [i * width + j] - check_data [i * width + j]) > max_diff) {
                    max_diff = abs(data [i * width + j] - check_data [i * width + j]);
                }
            }
        }
        cout << "max diff: " << max_diff << endl;
    }

};

int main() {
    HeatConduction heatConduction(0.1, 10000, 10000);
    heatConduction.loadDataFromFile("data.txt");
    heatConduction.save2img("heatConduction.bmp");
    heatConduction.cpuSolver(5);
    heatConduction.save2img("heatConduction_cpu.bmp");
    heatConduction.gpuSolver(5);
    heatConduction.save2img("heatConduction_gpu.bmp");
    heatConduction.check();
    return 0;
}