#include<iostream>
#include<string>
#include<cstring>
#include<chrono>

// #include "mpi.h"
// #include "cuda.h"
// #include "cuda_runtime_api.h"
using namespace std;
float K;
float *data,*tmp_data;
int iterations;
int width,height;
chrono::time_point<chrono::system_clock> startTime;
void draw_circle(float* data,int width,int height,float x,float y,float r,float val){
    for(int i=0;i<width;i++){
        for(int j=0;j<height;j++){
            float dx=i-x;
            float dy=j-y;
            if(dx*dx+dy*dy<r*r){
                data[i*height+j]=1;
            }
        }
    }
}
//prepare 数据准备,内存申请
void  prepare(){
    data =  (float*)malloc(sizeof(float)*width*height);
    tmp_data = (float*)malloc(sizeof(float)*width*height);
    memset(data,0,sizeof(float)*width*height);
    memset(tmp_data,0,sizeof(float)*width*height);
    draw_circle(data,width,height,width/2,height/2,width/2,255);
}
void singleCPUSolver(){
    for (int i = 0;i < height;i++) {
            for (int j = 0;j < width;j++) {
                //边界条件
                if (i == 0 || i == height - 1 || j == 0 || j == width - 1) {
                    tmp_data [i * width + j] = data [i * width + j];
                }
                else {
                    double delta = (K) * (data [(i - 1) * width + j] + data [(i + 1) * width + j] + data [i * width + j - 1] + data [i * width + j + 1] - 4 * data [i * width + j]);
                    tmp_data [i * width + j] = data [i * width + j] + delta;
                }
            }
        }
    //更新新的数据
    memcpy(data,tmp_data,sizeof(float)*width*height);
}
void finiaze(){
    free(data);
    free(tmp_data);
}
void prase_argv(int argc, char *argv[]){
    if(argc!=5){
        cout<<"Usage: "<<argv[0]<<" <K> <iterations> <width> <height>"<<endl;
        exit(1);
    }
    K=atof(argv[1]);
    iterations=atoi(argv[2]);
    width=atoi(argv[3]);
    height=atoi(argv[4]);
}
void startTimeCounter(chrono::time_point<chrono::system_clock> &startTime){
    startTime = chrono::system_clock::now();
}
void endTimeCounter(chrono::time_point<chrono::system_clock> &startTime){
    auto endTime = chrono::system_clock::now();
    printf("Time: %f ms\n", chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count()/1.0);
}
int main(int argc, char *argv[]) {
    prase_argv(argc, argv);
    prepare();
    startTimeCounter(startTime);
    for(int i=0;i<iterations;i++){
        singleCPUSolver();
    }
    endTimeCounter(startTime);
    finiaze();
    return 0;
}