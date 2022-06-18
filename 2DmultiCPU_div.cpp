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
float* init_data;
float *halo_buf[6];
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
int local_height;
int local_width;

void startTimeCounter(chrono::time_point<chrono::system_clock> &startTime){
    startTime = chrono::system_clock::now();
}
float endTimeCounter(chrono::time_point<chrono::system_clock> &startTime){
    auto endTime = chrono::system_clock::now();
    return chrono::duration_cast<chrono::microseconds>(endTime - startTime).count()/1000.0;
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
    return res;
}
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
void BrocastInitState(float* init_data,int width,int height){
    float *buffer;
    int *sendcounts;
    if(my_rank==0){
        int index =0;
        float *buffer = (float*)malloc(sizeof(float)*width*height);
        int *sendcounts = (int*)malloc(sizeof(int)*my_size);
        for(int i=0;i<my_size;i++){
            int x = i%grid_x;
            int y = i/grid_x;
            int x_start = x*width/grid_x;
            int x_end = (x+1)*width/grid_x;
            int y_start = y*height/grid_y;
            int y_end = (y+1)*height/grid_y;
            int local_height = y_end-y_start;
            int local_width = x_end-x_start;
            sendcounts[i] = local_width*local_height;
            for(int i = x_start;i<x_end;i++){
                for(int j = y_start;j<y_end;j++){
                    buffer[index++] = init_data[i*height+j];
                }
            }
        }
    }
    // MPI_Scatterv(buffer,sendcounts,NULL,MPI_FLOAT,data,local_width*local_height,MPI_FLOAT,0,MPI_COMM_WORLD);
    free(buffer);
    free(sendcounts);
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
    local_height = y_end-y_start;
    local_width = x_end-x_start;
    printf("my_rank:%d,grid_x:%d,grid_y:%d,x:%d,y:%d,x_start:%d,x_end:%d,y_start:%d,y_end:%d\n",my_rank,grid_x,grid_y,x,y,x_start,x_end,y_start,y_end);
    halo_buf[0] = (float*)malloc(sizeof(float)*local_width);
    halo_buf[1] = (float*)malloc(sizeof(float)*local_height);
    halo_buf[2] = (float*)malloc(sizeof(float)*local_width);
    halo_buf[3] = (float*)malloc(sizeof(float)*local_height);
    halo_buf[4] = (float*)malloc(sizeof(float)*local_height);
    halo_buf[5] = (float*)malloc(sizeof(float)*local_height);
    if(my_rank == 0){
        init_data = (float*)malloc(sizeof(float)*width*height);
        memset(init_data,0,sizeof(float)*width*height);
    }
    data =  (float*)malloc(sizeof(float)*local_width*local_height);
    tmp_data = (float*)malloc(sizeof(float)*local_width*local_height);
    memset(data,0,sizeof(float)*local_width*local_height);
    memset(tmp_data,0,sizeof(float)*local_width*local_height);
    if(my_rank==0){
        draw_circle(init_data,width,height,width/2,height/2,width/2,255);
    }

    BrocastInitState(init_data,width,height);

}

void HaloExchange(){
    int neighbor_left = (x-1+grid_x)%grid_x;
    int neighbor_right = (x+1)%grid_x;
    int neighbor_up = (y-1+grid_y)%grid_y;
    int neighbor_down = (y+1)%grid_y;
    for(int i=0;i<local_height;i++){
        halo_buf[4][i] = data[i*local_width];//left self trans
        halo_buf[5][i] = data[(i+1)*local_width-1];//right self trans
    }
    if(y%2 == 0){
        MPI_Sendrecv(data,local_width,MPI_FLOAT,neighbor_up,0,halo_buf[0],local_width,MPI_FLOAT,neighbor_up,0,MPI_COMM_WORLD,NULL);
    }else{
        MPI_Sendrecv(data+(local_height-1)*local_width,local_width,MPI_FLOAT,neighbor_down,0,halo_buf[2],local_width,MPI_FLOAT,neighbor_down,0,MPI_COMM_WORLD,NULL);
    } 
    if(y%2 == 0){
        MPI_Sendrecv(data+(local_height-1)*local_width,local_width,MPI_FLOAT,neighbor_down,1,halo_buf[2],local_width,MPI_FLOAT,neighbor_down,1,MPI_COMM_WORLD,NULL);
    }else{
        MPI_Sendrecv(data,local_width,MPI_FLOAT,neighbor_up,1,halo_buf[0],local_width,MPI_FLOAT,neighbor_up,1,MPI_COMM_WORLD,NULL);
    }
    if(x%2 == 0){
        MPI_Sendrecv(halo_buf[4],local_height,MPI_FLOAT,neighbor_left,2,halo_buf[3],local_height,MPI_FLOAT,neighbor_left,2,MPI_COMM_WORLD,NULL);
    }else{
        MPI_Sendrecv(halo_buf[5],local_height,MPI_FLOAT,neighbor_right,2,halo_buf[1],local_height,MPI_FLOAT,neighbor_right,2,MPI_COMM_WORLD,NULL);
    }
    if(x%2 == 0){
        MPI_Sendrecv(halo_buf[5],local_height,MPI_FLOAT,neighbor_right,3,halo_buf[1],local_height,MPI_FLOAT,neighbor_right,3,MPI_COMM_WORLD,NULL);
    }else{
        MPI_Sendrecv(halo_buf[4],local_height,MPI_FLOAT,neighbor_left,3,halo_buf[3],local_height,MPI_FLOAT,neighbor_left,3,MPI_COMM_WORLD,NULL);
    }
}
void calculate(){
    for(int i=0;i<local_height;i++){
        for(int j=0;j<local_width;j++){
            float delta;
            float up,down,left,right;
            if(i==0){
                up = halo_buf[0][i];
            }else{
                up = data[(i-1)*local_width+j];
            }
            if(i == local_height-1){
                down = halo_buf[2][i];
            }
            else{
                down = data[(i+1)*local_width+j];
            }
            if(j==0){
                left = halo_buf[3][i];
            }else{
                left = data[i*local_width+j-1];
            }
            if(j == local_width-1){
                right = halo_buf[1][i];
            }else{
                right = data[i*local_width+j+1];
            }
            delta = (K) * (up+down+left+right - 4 * data [i * local_width + j]);
            tmp_data[i*local_width+j] = data[i*local_width+j] + delta;
        }
    }
    memcpy(data,tmp_data,sizeof(float)*local_width*local_height);
}
void multiCPUSolverHalo(){
    auto time = chrono::system_clock::now();
    HaloExchange();
    dataTransferTime += endTimeCounter(time);
    calculate();
}
void finiaze(){
    free(data);
    free(tmp_data);
    free(init_data);
    for(int i=0;i<6;i++){
        free(halo_buf[i]);
    }
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
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    prase_argv(argc, argv);
    prepare();
    MPI_Barrier(MPI_COMM_WORLD);
    startTimeCounter(startTime);
    for(int i=0;i<iterations;i++){
        multiCPUSolverHalo();
    }
    float time = endTimeCounter(startTime);
    if(my_rank ==0){
        printf("dataTransferTime:%f ms\n",dataTransferTime);
        printf("time:%f ms\n",time);
    }
    finiaze();
    MPI_Finalize();
    return 0;
}