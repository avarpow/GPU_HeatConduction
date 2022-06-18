#include<iostream>
#include<string>
#include<cstring>
#include<chrono>

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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
// #include "mpi.h"
// #include "cuda.h"
// #include "cuda_runtime_api.h"
using namespace std;
float K;
float *data,*tmp_data;
float *device_data,*device_tmp_data;
int iterations;
int width,height;
chrono::time_point<chrono::system_clock> startTime;
__global__ void kernel(float *data,float *tmp_data,int width,int height,float K)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<width && y<height){
        if (x == 0 || x == height - 1 || y == 0 || y == width - 1) {
            tmp_data [x * width + y] = data [x * width + y];
        }
        else {
            double delta = (K) * (data [(x - 1) * width + y] + data [(x + 1) * width + y] + data [x * width + y - 1] + data [x * width + y + 1] - 4 * data [x * width + y]);
            tmp_data [x * width + y] = data [x * width + y] + delta;
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
    data =  (float*)malloc(sizeof(float)*width*height);
    tmp_data = (float*)malloc(sizeof(float)*width*height);
    gpuErrchk(cudaMalloc(&device_data,sizeof(float)*width*height));
    gpuErrchk(cudaMalloc(&device_tmp_data,sizeof(float)*width*height));
    memset(data,0,sizeof(float)*width*height);
    memset(tmp_data,0,sizeof(float)*width*height);
    draw_circle(data,width,height,width/2,height/2,width/2,255);
    gpuErrchk(cudaMemcpy(device_data,data,sizeof(float)*width*height,cudaMemcpyHostToDevice));
}
void singleGPUSolver(){
    dim3 dimBlock(8,8);
    dim3 dimGrid(width/8+1,height/8+1);
    kernel<<<dimGrid,dimBlock>>>(device_data,device_tmp_data,width,height,K);
    //更新新的数据
    gpuErrchk(cudaMemcpy(device_data,device_tmp_data,sizeof(float)*width*height,cudaMemcpyDeviceToDevice));
}
void finiaze(){
    gpuErrchk(cudaMemcpy(data,device_data,sizeof(float)*width*height,cudaMemcpyDeviceToHost));
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
float endTimeCounter(chrono::time_point<chrono::system_clock> &startTime){
    auto endTime = chrono::system_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count() / 1000.0;
}
int main(int argc, char *argv[]) {
    prase_argv(argc, argv);
    startTimeCounter(startTime);
    prepare();
    float prepare_time = endTimeCounter(startTime);
    cout<<"prepare time:"<<prepare_time<<endl;
    startTimeCounter(startTime);
    for(int i=0;i<iterations;i++){
        singleGPUSolver();
    }
    float cal_time = endTimeCounter(startTime);
    cout<<"cal_time:"<<cal_time<<endl;
    finiaze();
    return 0;
}