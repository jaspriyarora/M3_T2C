#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <CL/cl.hpp>

#define NUM_POINTS 1000000
#define NUM_CLUSTERS 10
#define DIMENSIONS 3
#define WORKGROUP_SIZE 256
#define LOCAL_SIZE WORKGROUP_SIZE

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize the random seed using the current time
    srand(time(NULL) + rank);

    // Determine the local portion of the data to be clustered
    int local_size = NUM_POINTS / size;
    float local_data[local_size * DIMENSIONS];
    for (int i = 0; i < local_size; i++)
    {
        for (int j = 0; j < DIMENSIONS; j++)
        {
            local_data[i * DIMENSIONS + j] = (float)rand() / RAND_MAX;
        }
    }

    // Create OpenCL context and command queue
    cl::Context context(CL_DEVICE_TYPE_GPU);
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::CommandQueue queue(context, devices[0]);

    // Create OpenCL buffers for the input and output arrays
    cl::Buffer data_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, local_size * DIMENSIONS * sizeof(float), local_data);
    cl::Buffer cluster_centers_buffer(context, CL_MEM_READ_WRITE, NUM_CLUSTERS * DIMENSIONS * sizeof(float));
    cl::Buffer assignments_buffer(context, CL_MEM_READ_WRITE, local_size * sizeof(int));

    // Initialize cluster centers randomly
    float cluster_centers[NUM_CLUSTERS * DIMENSIONS];
    if (rank == 0)
    {
        for (int i = 0; i < NUM_CLUSTERS; i++)
        {
            for (int j = 0; j < DIMENSIONS; j++)
            {
                cluster_centers[i * DIMENSIONS + j] = (float)rand() / RAND_MAX;
            }
        }
    }
    MPI_Bcast(cluster_centers, NUM_CLUSTERS * DIMENSIONS, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Create OpenCL program and kernel
    cl::Program program(context, "#define DIMENSIONS " + std::to_string(DIMENSIONS) + "\n\
        __kernel void assign(__global float *data, __global float *cluster_centers, __global int *assignments, const int local_size) { \n\
            int gid = get_global_id(0); \n\
            int lid = get_local_id(0); \n\
            int idx = gid * DIMENSIONS; \n\
            float min_distance = 1e9; \n\
            int min_cluster = -1; \n\
            for(int c = 0; c < NUM_CLUSTERS; c++) { \n\
                float distance = 0; \n\
                for(int i = 0; i < DIMENSIONS; i++) { \n\
                    distance += (data[idx + i] - cluster_centers[c * DIMENSIONS + i]) * (data[idx + i] - cluster_centers[c * DIMENSIONS + i]); \n\
                } \n\
                if(distance < min_distance) { \n\
                    min_distance = distance; \n\
                    min_cluster = c; \n\
                } \n\
            } \n\
            assignments[gid] = min_cluster; \n\
        } \
