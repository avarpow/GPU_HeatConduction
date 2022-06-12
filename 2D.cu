#include<iostream>
#include<string>
#include "EasyBMP.hpp"
#include "mpi.h"
using namespace std;
using namespace EasyBMP;


class HeatConduction {
public:
    double K;
    double height;
    double width;
    double* data;
    double range_max = 1000;
    HeatConduction(double K, double height, double width) {
        this->K = K;
        this->height = height;
        this->width = width;
        data = new double [(int)height * (int)width];
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
            Image img(width, height,filename,black);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    RGBColor color(data [i * width + j] / range_max, data [i * width + j] / range_max, data [i * width + j] / range_max);
                    color.r = 255-color.r;
                    color.g = 255-color.g;
                    color.b = 255-color.b;
                    img.SetPixel(j, i, color);
                }
            }
            img.Write();
        }
    }
    //cpu单线程
    void cpuSolver(int iteration) {
        double* new_data = new double [(int)height * (int)width];
        for(int k=0;k<iteration;k++){
            for(int i=0;i<height;i++){
                for(int j=0;j<width;j++){
                    if(i==0||i==height-1||j==0||j==width-1){
                        new_data[i*width+j]=data[i*width+j];
                    }
                    else{
                        double delta=(K)*(data[(i-1)*width+j]+data[(i+1)*width+j]+data[i*width+j-1]+data[i*width+j+1]-4*data[i*width+j]);
                        new_data[i*width+j]=data[i*width+j]+delta;
                    }
                }
            }
        }

    }
    //单gpu
    /*
    block size: 16*16
    each thread : 16*16

    */
    const row_per_thread = 16;
    const col_per_thread = 16;
    void gpuSolver(int iteration) {
        dim3 dimBlock(16, 16);
        dim3 dimGrid((int)width / row_per_thread / dimBlock.x + 1 , (int)height / scol_per_thread / dimBlock.y);
        double *d_data,*d_new_data;
    
    }
    //cpuMPI 实现
    void cpuMPIsolver(int iteration) {
        //TODO
    }
    //gpuMPI 实现
    
    void gpuMPIsolver(int iteration) {
        const int GPU_PER_NODE = 4;
        //TODO
    }

};
int main() {


    return 0;
}