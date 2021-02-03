#include "head.cuh"


void deviceProperty(void)
{
    int deviceCount = 0;
    int dev=0, driverVersion = 0, runtimeVersion = 0;

    CheckError(cudaGetDeviceCount(&deviceCount));
    if(deviceCount > 0){
        printf("Detected %d cuda capable devices\n", deviceCount);
    }else{
        printf("There is no avaliable device that support cuda\n");
        exit(-1);
    }

    CheckError(cudaSetDevice(dev));
    
    cudaDeviceProp deviceProp;
    CheckError(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device name: \"%s\"\n", deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA driver version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("CUDA runtime version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    printf("CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);

    float totalGobalMemory_MB = deviceProp.totalGlobalMem / (1024*1024);
    long long unsigned int totalGobalMemory_bytes = deviceProp.totalGlobalMem; 
    printf("Total Global Memory: %.0f MBytes, %llu bytes\n", totalGobalMemory_MB, totalGobalMemory_bytes);

    int Multiporcessors = deviceProp.multiProcessorCount;
    int CudaCores_per_MP = 1; //_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
    int CudaCores = 1; //CudaCores_per_MP * Multiporcessors;
    printf("Multiporcessors: %2d ", Multiporcessors);

    long long unsigned int sharedMemPerBlock = deviceProp.sharedMemPerBlock;
    long long unsigned int regsPerBlock = deviceProp.regsPerBlock;
    long long unsigned int memPitch;
    long long unsigned int totalConstMem = deviceProp.totalConstMem;
    long long unsigned int textureAlignment = deviceProp.textureAlignment;

    printf("sharedMemPerBlock: %zu bytes\n", sharedMemPerBlock);
    printf("regsPerBlock: %d bytes\n", regsPerBlock);
    printf("memPitch: %zu bytes\n", memPitch);
    printf("totalConstMem: %zu bytes\n", totalConstMem);
    printf("textureAlignment: %zu bytes\n", textureAlignment);

    int warpSize = deviceProp.warpSize;
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor = deviceProp.maxThreadsPerMultiProcessor;
    int clockRate = deviceProp.clockRate * 1e-3f;
    int memoryClockRate = deviceProp.memoryClockRate * 1e-3f;
    int memoryBusWidth = deviceProp.memoryBusWidth;
    long long unsigned int l2CacheSize = deviceProp.l2CacheSize;

    printf("maxThreadsPerBlock: %zu bytes\n", maxThreadsPerBlock);
    printf("warpSize: %d bytes\n", warpSize);
    printf("maxThreadsPerMultiProcessor: %d bytes\n", maxThreadsPerMultiProcessor);
    printf("clockRate: %f MB bytes\n", clockRate);
    printf("memoryClockRate: %f MB bytes\n", memoryClockRate);
    printf("memoryBusWidth: %d bytes\n", memoryBusWidth);
    printf("l2CacheSize: %d bytes\n", l2CacheSize);

    int maxThreadsDim[3];
    maxThreadsDim[0] = deviceProp.maxThreadsDim[0];
    maxThreadsDim[1] = deviceProp.maxThreadsDim[1];
    maxThreadsDim[2] = deviceProp.maxThreadsDim[2];
    int maxGridSize[3];
    maxGridSize[0] = deviceProp.maxGridSize[0];
    maxGridSize[1] = deviceProp.maxGridSize[1];
    maxGridSize[2] = deviceProp.maxGridSize[2];

    printf("maxThreadsDim(x,y,z): (%d, %d, %d)\n", maxThreadsDim[0], maxThreadsDim[1], maxThreadsDim[3]);
    printf("maxGridSize(x,y,z): (%d, %d, %d)\n", maxGridSize[0], maxGridSize[1], maxGridSize[3]);

    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~common use: cudaChooseDevice~~~~~~~~~~~~~~~~~~~~~~~*/
    printf("\n\nplease choose a GPU which satisfied the prop\n");
    cudaDeviceProp prop1;
    int dev1;
    cudaGetDevice(&dev1);
    printf("ID of current CUDA device: %d\n", dev1);
    
    memset(&prop1, 0, sizeof(prop1));
    prop1.major=6;
    prop1.minor=1;
    cudaChooseDevice(&dev1, &prop1);
    printf("ID of CUDA device closet to revision 6.1: %d\n", dev1);

    cudaGetDeviceProperties(&prop1, dev);
    printf("NAME of CUDA device closet to revision 6.1: \"%s\"\n\n\n", prop1.name);
    cudaSetDevice(dev1);
}

__global__ void simpleAssert_kernel(int size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    assert( x < size);
}
void simpleAssert()
{
    bool result = true;

    dim3 block(32);
    dim3 grid(2);

    printf("Launch kernel to generate assertion failures\n");
    simpleAssert_kernel<<<grid, block>>>(60);
    printf("\n-- Being assert output\n\n");
    cudaError_t error = cudaDeviceSynchronize();
    printf("\n-- End assert output\n\n");

    if(error == cudaErrorAssert)
    {
        printf("Device assert failed as expected, error message is %s\n\n", cudaGetErrorString(error));
    }

    result = error == cudaErrorAssert;
    printf("simpleAssert result: %s\n\n\n\n", result ? "true" : "false");

}


__global__ void simpleAtomic_kernel(int * d_data)
{
    const int tx = blockDim.x * blockIdx.x + threadIdx.x;

    //if(tx == 0)printf("d_data[0]=%d, d_data[1]=%d\n",d_data[0],d_data[1]);
    atomicAdd(&d_data[0], 10);
    //printf("tx=%d, d_data[0]=%d\n",tx, d_data[0]);
    atomicSub(&d_data[1], 10);
    //if(tx == 0)printf("d_data[1]=%d\n",d_data[1]);
    //d_data[0] += 10;
    if(tx == 1) atomicExch(&d_data[2], tx);
    if(tx == 20) atomicMax(&d_data[3], tx);
    if(tx == 3) atomicMin(&d_data[4], tx);
    atomicInc((unsigned int *)&d_data[5], 16); //atomic increment
    atomicDec((unsigned int *)&d_data[6], 137);
    atomicCAS(&d_data[7], tx-1, tx);
    atomicAnd(&d_data[8], 2*tx+7);
    atomicOr(&d_data[9], 1 << tx);
    atomicXor(&d_data[10], tx);

}
void simpleAtomicIntrinsics()
{

    int dataSize = 11;
    int memSize = dataSize * sizeof(int);
    
    int *h_data; //(int *) malloc(memSize);
    cudaMallocHost((void**)&h_data, memSize);

    for(int i=0; i < dataSize; i++)
        h_data[i] = 1;
    
    h_data[8] = h_data[10] = 0xff;

    cudaStream_t stream;
    CheckError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int *d_data;
    CheckError(cudaMalloc((void**)&d_data, memSize));
    CheckError(cudaMemcpyAsync(d_data, h_data, memSize, cudaMemcpyHostToDevice, stream));

    dim3 block(32);
    dim3 grid(10);
    simpleAtomic_kernel<<<grid, block, 0, stream>>>(d_data);
    
    CheckError(cudaMemcpyAsync(h_data, d_data, memSize, cudaMemcpyDeviceToHost,stream));
    CheckError(cudaStreamSynchronize(stream));

    for(int i=0; i< dataSize; i++)
    {
        printf("h_data[%d]=%d\n", i, h_data[i]);
    }
    printf("h_data[0]:%d, h_data[1]=%d\n", h_data[0],h_data[1]);

    //free(h_data);
    cudaFreeHost(h_data);
    cudaFree(d_data);

}

int main()
{
 
    //deviceProperty();
    //simpleAssert();
    simpleAtomicIntrinsics();


    return 0;
}
