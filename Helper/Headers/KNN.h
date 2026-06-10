#pragma once
#include <CL/cl.h>
#include <stdio.h>
#include "globals.h"

cl_mem train_buf;
cl_mem dist_buf;

const char* knn_kernel = R"(
__kernel void knn_distance(
    __global const int* training,
    __global const int* query,
    __global float* distances,
    const int n_attrs,
    const int n_samples
){
    int id = get_global_id(0);
    if(id >= n_samples) return;

    float dist = 0.0f;
    float weights[6] = {1.0f, 1.0f, 1.0f, 1.0f, 3.0f, 1.0f};
    for(int i = 0; i < n_attrs; i++){
        if(training[id * n_attrs + i] != query[i]) dist += weights[i];
    }
    distances[id] = dist;
})";



extern "C" void init_knn(int* training, int n_attrs, int n_samples){
    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, &knn_kernel, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if(err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Build error: %s\n", log);
        return;
    }
    kernel = clCreateKernel(program, "knn_distance", NULL);

    train_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*n_samples*n_attrs, training, NULL);
    dist_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*n_samples, NULL, NULL);
}

extern "C" void knn_distances(int* query, float* distances, int n_attrs, int n_samples){
    cl_mem query_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*n_attrs, query, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &train_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &query_buf);  
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &dist_buf);  
    clSetKernelArg(kernel, 3, sizeof(int), &n_attrs);  
    clSetKernelArg(kernel, 4, sizeof(int), &n_samples);
    
    size_t global_size = n_samples;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueReadBuffer(queue, dist_buf, CL_TRUE, 0, sizeof(float)*n_samples, distances, 0, NULL, NULL);
}