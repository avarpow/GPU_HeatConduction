#include<iostream>
#include<string>
#include<cstring>
#include<chrono>

#include "mpi.h"
// #include "cuda.h"
// #include "cuda_runtime_api.h"
using namespace std;
float K;
float *data,*tmp_data;
int iterations;
int width,height;
chrono::time_point<chrono::system_clock> startTime;
float dataTransferTime=0;
int my_rank,my_size;
int grid_x;
int grid_y;
int x;
int y;
int x_start;
int x_end;
int y_start;
int y_end;

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
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&my_size);
    grid_x = calGridSize(my_size);
    grid_y = my_size/grid_x;
    x = my_rank%grid_x;
    y = my_rank/grid_x;
    x_start = x*width/grid_x;
    x_end = (x+1)*width/grid_x;
    y_start = y*height/grid_y;
    y_end = (y+1)*height/grid_y;

    printf("my_rank:%d,grid_x:%d,grid_y:%d,x:%d,y:%d,x_start:%d,x_end:%d,y_start:%d,y_end:%d\n",my_rank,grid_x,grid_y,x,y,x_start,x_end,y_start,y_end);
    data =  (float*)malloc(sizeof(float)*width*height);
    tmp_data = (float*)malloc(sizeof(float)*width*height);
    memset(data,0,sizeof(float)*width*height);
    memset(tmp_data,0,sizeof(float)*width*height);
    if(my_rank==0){
        draw_circle(data,width,height,width/2,height/2,width/2,255);
    }
    MPI_Bcast(data,width*height,MPI_FLOAT,0,MPI_COMM_WORLD);
}
int calGridSize(int size){
    int min =999;
    int res =-1;
    for(int i=1;i<=size;i++){
        if(size%i==0){
            if((i+size/i)<min){
                min=(i+size/i);
                res=i;
            }
        }
    }
    return 0;
}
void multiCPUSolverReduce(){
    for (int i = y_start;i < y_end;i++) {
            for (int j = x_start;j < x_end;j++) {
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
    auto time = chrono::system_clock::now();
    MPI_Allreduce(tmp_data,data,width*height,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    dataTransferTime += endTimeCounter(time);
}
void finiaze(){
    free(data);
    free(tmp_data);
}
void prase_argv(int argc, char *argv[]){
    if(argc!=4){
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
float endTimeCounter(chrono::time_point<chrono::system_clock> &startTime){
    auto endTime = chrono::system_clock::now();
}
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    prase_argv(argc, argv);
    prepare();
    startTimeCounter(startTime);
    multiCPUSolverReduce();
    float time = endTimeCounter(startTime);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank==0){
        cout<<"Time: "<<time<<endl;
        cout<<"Data Transfer Time: "<<dataTransferTime<<endl;
    }
    finiaze();
    return 0;
}