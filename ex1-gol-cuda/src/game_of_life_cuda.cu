#include "common.h"

#define INCLUDE_CPU_VERSION
#include "game_of_life.c"

void __cuda_check_error(cudaError_t err, const char *file, int line){
	if(err != cudaSuccess){
        fprintf(stderr, "CUDA error (%s:%d): %s\n", file, line, cudaGetErrorString(err));
        exit(1);
    }
}

#define CUDA_CHECK_ERROR(X)({\
	__cuda_check_error((X), __FILE__, __LINE__);\
})

#define MALLOC_CHECK_ERROR(X)({\
    if ((X) == 0){\
        fprintf(stderr, "Malloc error (%s:%d): %i\n", __FILE__, __LINE__, (X));\
        exit(1);\
    }\
})

__global__ void gpu_game_of_life_step(int *current_grid, int *next_grid, int n, int m){
    int neighbours;
    int n_i[8], n_j[8];

    // get the unique index (turn into i, j coordinates) of the thread to operate
    // on the n x m matrix
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int i = idx / m;
    unsigned int j = idx % m;

    // count the number of neighbours, clockwise around the current cell.
    neighbours = 0;
    n_i[0] = i - 1; n_j[0] = j - 1;
    n_i[1] = i - 1; n_j[1] = j;
    n_i[2] = i - 1; n_j[2] = j + 1;
    n_i[3] = i;     n_j[3] = j + 1;
    n_i[4] = i + 1; n_j[4] = j + 1;
    n_i[5] = i + 1; n_j[5] = j;
    n_i[6] = i + 1; n_j[6] = j - 1;
    n_i[7] = i;     n_j[7] = j - 1;

    if(n_i[0] >= 0 && n_j[0] >= 0 && current_grid[n_i[0] * m + n_j[0]] == ALIVE) neighbours++;
    if(n_i[1] >= 0 && current_grid[n_i[1] * m + n_j[1]] == ALIVE) neighbours++;
    if(n_i[2] >= 0 && n_j[2] < m && current_grid[n_i[2] * m + n_j[2]] == ALIVE) neighbours++;
    if(n_j[3] < m && current_grid[n_i[3] * m + n_j[3]] == ALIVE) neighbours++;
    if(n_i[4] < n && n_j[4] < m && current_grid[n_i[4] * m + n_j[4]] == ALIVE) neighbours++;
    if(n_i[5] < n && current_grid[n_i[5] * m + n_j[5]] == ALIVE) neighbours++;
    if(n_i[6] < n && n_j[6] >= 0 && current_grid[n_i[6] * m + n_j[6]] == ALIVE) neighbours++;
    if(n_j[7] >= 0 && current_grid[n_i[7] * m + n_j[7]] == ALIVE) neighbours++;

    if(current_grid[i*m + j] == ALIVE && (neighbours == 2 || neighbours == 3)){
        next_grid[i*m + j] = ALIVE;
    } else if(current_grid[i*m + j] == DEAD && neighbours == 3){
        next_grid[i*m + j] = ALIVE;
    } else {
        next_grid[i*m + j] = DEAD;
    }
}


/*
Implements the game of life on a grid of size `n` times `m`, starting from the `initial_state` configuration.

If `nsteps` is positive, returns the last state reached.
*/
int* gpu_game_of_life(const int *initial_state, int n, int m, int nsteps, float *kernel_time){
    struct timeval start;
    *kernel_time = 0.0;

    unsigned int nThreadsPerBlock, nBlocks;
    int *grid = (int *) malloc(sizeof(int) * n * m);

    int current_step = 0;
    int *tmp = NULL;

    int *dev_grid, *dev_updated_grid;
    CUDA_CHECK_ERROR(cudaMalloc(&dev_grid, sizeof(int) * n * m));
    CUDA_CHECK_ERROR(cudaMalloc(&dev_updated_grid, sizeof(int) * n * m));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_grid, initial_state, sizeof(int) * n * m, cudaMemcpyHostToDevice));

    if (n*m > 1024) { 
		nThreadsPerBlock = 1024;
	} else {
		nThreadsPerBlock = n*m;
	}
	nBlocks = (n*m + nThreadsPerBlock - 1) / nThreadsPerBlock;

    while(current_step != nsteps){
        current_step++;

        // Uncomment the following 2 lines if you want to print the state at every step
        // CUDA_CHECK_ERROR(cudaMemcpy(grid, dev_grid, sizeof(float) * n * m, cudaMemcpyDeviceToHost));
        // visualise(VISUAL_ASCII, current_step, grid, n, m);
        start = init_time();
        gpu_game_of_life_step<<<nBlocks, nThreadsPerBlock>>>(dev_grid, dev_updated_grid, n, m);
        cudaDeviceSynchronize();
        *kernel_time += get_elapsed_time(start);
        // swap current and updated grid
        tmp = dev_grid;
        dev_grid = dev_updated_grid;
        dev_updated_grid = tmp;
    }
    // Copy result back to host and free device arrays
    CUDA_CHECK_ERROR(cudaMemcpy(grid, dev_grid, sizeof(float) * n * m, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(dev_grid));
    CUDA_CHECK_ERROR(cudaFree(dev_updated_grid));

    return grid;
}

int main(int argc, char **argv)
{


    struct Options *opt = (struct Options *) malloc(sizeof(struct Options));
    getinput(argc, argv, opt);
    int n = opt->n, m = opt->m, nsteps = opt->nsteps;
    int *initial_state = (int *) malloc(sizeof(int) * n * m);
    if(!initial_state){
        printf("Error while allocating memory.\n");
        return -1;
    }
    generate_IC(opt->iictype, initial_state, n, m);
    struct timeval start;
    float kernel_time;

    // Run CPU version
    start = init_time();
    int *cpu_final_state = cpu_game_of_life(initial_state, n, m, nsteps);
    float cpu_elapsed = get_elapsed_time(start);

    // Run GPU version
    start = init_time();
    int *gpu_final_state = gpu_game_of_life(initial_state, n, m, nsteps, &kernel_time);
    float gpu_elapsed = get_elapsed_time(start);


    // Check correctness
    long correct = 0;
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            if(cpu_final_state[i*m + j] == gpu_final_state[i*m + j]) {
                correct++;
            }
        }
    }

    printf("CPU:        %f ms\n", cpu_elapsed);
    printf("GPU:        %f ms\n", gpu_elapsed);
    printf("Speedup:    %.2fx\n", cpu_elapsed / gpu_elapsed);
    printf("Kernel:     %f ms\n", kernel_time);
    printf("            %.3f%\n", kernel_time / gpu_elapsed * 100);
    
    // Print number of correct values (with color using escape codes)
    printf("Correct:    ");
    if (correct == n*m) { printf("\033[0;32m");} else { printf("\033[0;31m");}
    printf("%ld/%ld\n", correct, n*m);
    printf("\033[0m");
    
    FILE *fp = fopen("performance.txt", "a");
    fprintf(fp, "%d,%d,%d,%f,%f,%f,%f\n", n, m, nsteps, cpu_elapsed, gpu_elapsed, cpu_elapsed / gpu_elapsed, 
        kernel_time / gpu_elapsed * 100);
    fclose(fp);




        
    free(cpu_final_state);
    free(gpu_final_state);
    free(initial_state);
    free(opt);
    return 0;
}