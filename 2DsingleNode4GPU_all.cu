#include<iostream>
#include<string>
#include<cstring>
#include<chrono>

#include "mpi.h"
#include "cuda.h"
#include "nccl.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
using namespace std;
#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
float K;
float *data,*tmp_data;
float *device_data,*device_tmp_data;
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
ncclUniqueId id;
ncclComm_t nccl_comm;
cudaStream_t s;

__global__ void kernel(float *data,float *tmp_data,int x_start,int x_end,int y_start,int y_end,float K)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x+x_start;
    int y=blockIdx.y*blockDim.y+threadIdx.y+y_start;
    if(x<width && y<height){
        if(x==0||x == width||y == 0||y == height){
            tmp_data[y*width+x]= data[y*width+x];
        }
        else{
            float delta = (K) * (data [(i - 1) * width + j] + data [(i + 1) * width + j] + data [i * width + j - 1] + data [i * width + j + 1] - 4 * data [i * width + j]);
            tmp_data [i * width + j] = data [i * width + j] + delta;
        }
    }
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
//prepare 数据准备,内存申请
void  prepare(){
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&my_size);
    if (my_rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm, my_size, id, my_rank);
    if(my_size!=4){
        cout<<"error:only support 4 GPU"<<endl;
        exit(0);
    }
    
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
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaStreamCreate(&s));

    cudaMalloc(&device_data,sizeof(float)*width*height);
    cudaMalloc(&device_tmp_data,sizeof(float)*width*height);
    cudaMemcpy(device_data,data,sizeof(float)*width*height,cudaMemcpyHostToDevice);
    // MPI_Bcast(data,width*height,MPI_FLOAT,0,MPI_COMM_WORLD);
    ncclBcast(device_data,width*height,ncclFloat,0,nccl_comm,s);
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
void finiaze(){
    cudaMemcpy(data,device_data,sizeof(float)*width*height,cudaMemcpyDeviceToHost);
    cudaFree(device_data);
    cudaFree(device_tmp_data);
    free(data);
    free(tmp_data);

}
void multiGPUSolverReduce(){
    dim3 dimBlock(8,8);
    dim3 dimGrid(width/2/8,height/2/8);
    kernel<<<dimGrid,dimBlock>>>(device_data,device_tmp_data,x_start,x_end,y_start,y_end,K);
    auto time = chrono::system_clock::now();
    // MPI_Allreduce(tmp_data,data,width*height,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    ncclAllReduce(device_tmp_data,device_data,width*height,ncclFloat,ncclSum,nccl_comm,NULL);
    dataTransferTime += endTimeCounter(time);
}
void startTimeCounter(chrono::time_point<chrono::system_clock> &startTime){
    startTime = chrono::system_clock::now();
}
float endTimeCounter(chrono::time_point<chrono::system_clock> &startTime){
    auto endTime = chrono::system_clock::now();
}

int main(int argc, char *argv[]) {
    prase_argv(argc, argv);
    prepare();
    startTimeCounter(startTime);
    multiGPUSolverReduce();
    endTimeCounter(startTime);
    finiaze();
    return 0;
}