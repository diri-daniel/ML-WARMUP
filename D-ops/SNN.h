#pragma once
#include <CL/cl.h>
#include <stdio.h>
#include "globals.h"

const char* snn_kernel = R"(
__kernel void snn_neuronOut(
    __global const float* training,
    __global const float* weights,
    __global const float* bias,
    __global float* neuronValues,
    const int batchLength,
    const int neuronLength,
    const int inputLength
    ){
    int r = get_global_id(0);
    int c = get_global_id(1);

    if(r >= batchLength || c >= neuronLength) return;

    float sum = 0.0f + bias[c]; 
    for(int i = 0; i < inputLength; i++){
        sum += training[r * inputLength + i] * weights[i * neuronLength + c];
    }

    neuronValues[r * neuronLength + c] = sum;

})";


extern "C" void init_snn(){
    cl_int err;
    program = clCreateProgramWithSource(context, 1, &snn_kernel, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if(err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Build error: %s\n", log);
        return;
    }
    kernel = clCreateKernel(program, "snn_neuronOut", NULL);
    printf("snn initialized\n");
}

extern "C" void snn_forward(
    float* training, 
    float* weights, 
    float* bias, 
    float* neuronValues, 
    int batchLength, 
    int neuronLength, 
    int inputLength
){
    cl_int err;
    cl_mem trainbuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*batchLength*inputLength, training, &err);
    
    cl_mem weightbuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*inputLength*neuronLength, weights, &err);
    
    cl_mem biasbuf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*neuronLength, bias, &err);
    

    cl_mem neuronbuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*batchLength*neuronLength, NULL, &err);
    

    // int lsize = 64;
    cl_int cl_batch = (cl_int)batchLength;
    cl_int cl_neuron = (cl_int)neuronLength;
    cl_int cl_input = (cl_int)inputLength;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &trainbuf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &weightbuf);  
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &biasbuf);  
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &neuronbuf); 
    // clSetKernelArg(kernel, 4, sizeof(float)*lsize, NULL);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &cl_batch);  
    clSetKernelArg(kernel, 5, sizeof(cl_int), &cl_neuron);
    clSetKernelArg(kernel, 6, sizeof(cl_int), &cl_input);

    
    size_t global_size[2] = {(size_t)batchLength, (size_t)neuronLength};
    // size_t local_size[2] = {1, (size_t)lsize};
    float zero = 0.0f;
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
    
    clFinish(queue);
    
    clEnqueueReadBuffer(queue, neuronbuf, CL_TRUE, 0, sizeof(float)*batchLength*neuronLength, neuronValues, 0, NULL, NULL);
    
    clReleaseMemObject(trainbuf);
    clReleaseMemObject(biasbuf);
    clReleaseMemObject(weightbuf);
    clReleaseMemObject(neuronbuf);
    // for(int i = 0; i < 10; i++) printf("neuronbuf[%d] = %f\n", i, neuronValues[i]);
}
