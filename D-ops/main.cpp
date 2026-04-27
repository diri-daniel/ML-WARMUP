#include <CL/cl.h>
#include <stdio.h>
#include "KNN.h"
#include "SNN.h"

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_kernel kernel;
cl_program program;

extern "C" void init_opencl(){
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueueWithProperties(context, device , 0, NULL);
};

extern "C" void cleanup(){
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    printf("done");
}

int main() {
    return 0;
}
