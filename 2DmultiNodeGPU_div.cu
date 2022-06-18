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
float* init_data;
float *device_data,*device_tmp_data;
float *device_halo_buf[6];

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
ncclUniqueId id;
ncclComm_t nccl_comm;
cudaStream_t s;

__global__ void kernel(float *data,float *tmp_data,float *halo_buf[6],int local_width,int local_height,float k)
{
    int x=threadIdx.x+blockIdx.x*blockDim.x;
    int y=threadIdx.y+blockIdx.y*blockDim.y;
    float delta;
    if(x==0){
        delta = (K) * (halo_buf[0][y] + data [(x + 1) * local_width + y] + data [x * local_width + y - 1] + data [x * local_width + y + 1] - 4 * data [x * local_width + y]);
    }
    else if(x == local_height-1){
        delta = (K) * (halo_buf[1][y] + data [(x + 1) * local_width + y] + data [x * local_width + y - 1] + data [x * local_width + y + 1] - 4 * data [x * local_width + y]);
    }
    else if(y==0){
        delta = (K) * (halo_buf[3][y] + data [(x + 1) * local_width + y] + data [x * local_width + y - 1] + data [x * local_width + y + 1] - 4 * data [x * local_width + y]);
    }
    else if(y == local_width-1){
        delta = (K) * (halo_buf[2][y] + data [(x + 1) * local_width + y] + data [x * local_width + y - 1] + data [x * local_width + y + 1] - 4 * data [x * local_width + y]);
    }
    else{
        delta = (K) * (data [(x - 1) * local_width + y] + data [(x + 1) * local_width + y] + data [x * local_width + y - 1] + data [x * local_width + y + 1] - 4 * data [x * local_width + y]);
    }
    tmp_data[x*local_width+y] = data[x*local_width+y] + delta;

}
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
void BrocastInitState(float* init_data,int width,int height){
    float *buffer;
    int *sendcounts;
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
    }
    if(my_rank==0){
        int index =0;
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
    if(my_rank == 0){
        cudaMemcpy(device_data,buffer,sendcounts[0],cudaMemcpyHostToDevice);
        ncclSend(buffer+sendcounts[0],sendcounts[1],ncclFloat,1,nccl_comm,s);
        MPI_Send(buffer+sendcounts[0]+sendcounts[1],sendcounts[2],MPI_FLOAT,2,0,MPI_COMM_WORLD);
        MPI_Send(buffer+sendcounts[0]+sendcounts[1]+sendcounts[2],sendcounts[3],MPI_FLOAT,3,0,MPI_COMM_WORLD);
    }
    else if(my_rank == 1){
        ncclRecv(device_data,sendcounts[1],ncclFloat,0,nccl_comm,s);
    }
    else if(my_rank == 2 || my_rank == 3){
        MPI_Recv(data,sendcounts[my_rank],MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        cudaMemcpy(device_data,data,sendcounts[my_rank],cudaMemcpyHostToDevice);
    }
    // MPI_Scatterv(buffer,sendcounts,NULL,MPI_FLOAT,data,local_width*local_height,MPI_FLOAT,0,MPI_COMM_WORLD);
    
    free(buffer);
    free(sendcounts);
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
    local_height = y_end-y_start;
    local_width = x_end-x_start;
    printf("my_rank:%d,grid_x:%d,grid_y:%d,x:%d,y:%d,x_start:%d,x_end:%d,y_start:%d,y_end:%d\n",my_rank,grid_x,grid_y,x,y,x_start,x_end,y_start,y_end);
    data =  (float*)malloc(sizeof(float)*width*height);
    tmp_data = (float*)malloc(sizeof(float)*width*height);
    memset(data,0,sizeof(float)*width*height);
    memset(tmp_data,0,sizeof(float)*width*height);
    if(my_rank == 0){
        init_data = (float*)malloc(sizeof(float)*width*height);
        memset(init_data,0,sizeof(float)*width*height);

    }
    if(my_rank==0){
        draw_circle(init_data,width,height,width/2,height/2,width/2,255);
    }
    CUDACHECK(cudaSetDevice(localRank%2));
    CUDACHECK(cudaStreamCreate(&s));

    cudaMalloc(&device_halo_buf[0],sizeof(float)*local_width);
    cudaMalloc(&device_halo_buf[1],sizeof(float)*local_height);
    cudaMalloc(&device_halo_buf[2],sizeof(float)*local_width);
    cudaMalloc(&device_halo_buf[3],sizeof(float)*local_height);
    cudaMalloc(&device_halo_buf[4],sizeof(float)*local_height);
    cudaMalloc(&device_halo_buf[5],sizeof(float)*local_height);
    cudaMalloc(&device_data,sizeof(float)*width*height);
    cudaMalloc(&device_tmp_data,sizeof(float)*width*height);
    cudaMemcpy(device_data,data,sizeof(float)*width*height,cudaMemcpyHostToDevice);
    // MPI_Bcast(data,width*height,MPI_FLOAT,0,MPI_COMM_WORLD);
    BrocastInitState(init_data,width,height);
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
void HaloExchange(){
    int neighbor_left = (x-1+grid_x)%grid_x;
    int neighbor_right = (x+1)%grid_x;
    int neighbor_up = (y-1+grid_y)%grid_y;
    int neighbor_down = (y+1)%grid_y;
    for(int i=0;i<local_height;i++){
        halo_buf[4][i] = device_data[i*local_width];//left self trans
        halo_buf[5][i] = device_data[(i+1)*local_width-1];//right self trans
    }
    if(y%2 == 0){
        MPI_Sendrecv(device_data,local_width,MPI_FLOAT,neighbor_up,0,halo_buf[0],local_width,MPI_FLOAT,neighbor_up,0,MPI_COMM_WORLD,NULL);
    }else{
        MPI_Sendrecv(device_data+(local_height-1)*local_width,local_width,MPI_FLOAT,neighbor_down,0,halo_buf[2],local_width,MPI_FLOAT,neighbor_down,0,MPI_COMM_WORLD,NULL);
    } 
    if(y%2 == 0){
        MPI_Sendrecv(device_data+(local_height-1)*local_width,local_width,MPI_FLOAT,neighbor_down,1,halo_buf[2],local_width,MPI_FLOAT,neighbor_down,1,MPI_COMM_WORLD,NULL);
    }else{
        MPI_Sendrecv(device_data,local_width,MPI_FLOAT,neighbor_up,1,halo_buf[0],local_width,MPI_FLOAT,neighbor_up,1,MPI_COMM_WORLD,NULL);
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
    dim3 dimBlock(8,8);
    dim3 dimGrid(local_height,local_width);
    kernel<<<dimGrid,dimBlock>>>(device_data,device_tmp_data,halo_buf,local_width,local_height,K);
    cudaMemcpy(device_data,device_tmp_data,sizeof(float)*width*height,cudaMemcpyDeviceToDevice);
}
void multiCPUSolverHalo(){
    HaloExchange();
    calculate();
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