#include "nccl.h"
#include <chrono>
using namespace std;
extern float K;
extern float *data,*tmp_data;
extern float *device_data,*device_tmp_data;
extern int iterations;
extern int width,height;
extern float dataTransferTime;
extern int my_rank,my_size;
extern int grid_x;
extern int grid_y;
extern int x;
extern int y;
extern int x_start;
extern int x_end;
extern int y_start;
extern int y_end;
extern ncclUniqueId id;
extern ncclComm_t nccl_comm;
extern cudaStream_t s;
extern chrono::time_point<chrono::system_clock> startTime;
extern float endTimeCounter(chrono::time_point<chrono::system_clock> &startTime);

__global__ void kernel(float *data,float *tmp_data,int x_start,int x_end,int y_start,int y_end,int height,int width,float K)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x+x_start;
    int y=blockIdx.y*blockDim.y+threadIdx.y+y_start;
    if(x<width && y<height){
        if(x==0||x == width||y == 0||y == height){
            tmp_data[y*width+x]= data[y*width+x];
        }
        else{
            float delta = (K) * (data [(x - 1) * width + y] + data [(x + 1) * width + y] + data [x * width + y - 1] + data [x * width + y + 1] - 4 * data [x * width + y]);
            tmp_data [x * width + y] = data [x * width + y] + delta;
        }
    }
}

void multiGPUSolverReduce(){
    dim3 dimBlock(8,8);
    dim3 dimGrid(width/2/8,height/2/8);
    kernel<<<dimGrid,dimBlock>>>(device_data,device_tmp_data,x_start,x_end,y_start,y_end,height,width,K);
    auto time = chrono::system_clock::now();
    // MPI_Allreduce(tmp_data,data,width*height,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    ncclAllReduce(device_tmp_data,device_data,width*height,ncclFloat,ncclSum,nccl_comm,NULL);
    dataTransferTime += endTimeCounter(time);
}