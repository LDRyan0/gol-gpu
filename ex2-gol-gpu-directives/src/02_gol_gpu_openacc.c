#include "common.h"

#define N 10000
#define M 10000
#define NSTEPS 100

void game_of_life_cpu(struct Options *opt, int *current_grid, int *next_grid){
    int neighbours;
    int n_i[8], n_j[8];

    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
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

            if(n_i[0] >= 0 && n_j[0] >= 0 && current_grid[n_i[0] * M + n_j[0]] == ALIVE) neighbours++;
            if(n_i[1] >= 0 && current_grid[n_i[1] * M + n_j[1]] == ALIVE) neighbours++;
            if(n_i[2] >= 0 && n_j[2] < M && current_grid[n_i[2] * M + n_j[2]] == ALIVE) neighbours++;
            if(n_j[3] < M && current_grid[n_i[3] * M + n_j[3]] == ALIVE) neighbours++;
            if(n_i[4] < N && n_j[4] < M && current_grid[n_i[4] * M + n_j[4]] == ALIVE) neighbours++;
            if(n_i[5] < N && current_grid[n_i[5] * M + n_j[5]] == ALIVE) neighbours++;
            if(n_i[6] < N && n_j[6] >= 0 && current_grid[n_i[6] * M + n_j[6]] == ALIVE) neighbours++;
            if(n_j[7] >= 0 && current_grid[n_i[7] * M + n_j[7]] == ALIVE) neighbours++;

            if(current_grid[i*M + j] == ALIVE && (neighbours == 2 || neighbours == 3)){
                next_grid[i*M + j] = ALIVE;
            } else if(current_grid[i*M + j] == DEAD && neighbours == 3){
                next_grid[i*M + j] = ALIVE;
            }else{
                next_grid[i*M + j] = DEAD;
            }
        }
    }
}

void game_of_life_gpu(struct Options *opt, int *current_grid, int *next_grid){
    int neighbours;
    int n_i[8], n_j[8];

    #pragma acc parallel loop gang private(n_i, n_j)
    for(int i = 0; i < N; i++){
        #pragma acc loop vector
        for(int j = 0; j < M; j++){
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

            if(n_i[0] >= 0 && n_j[0] >= 0 && current_grid[n_i[0] * M + n_j[0]] == ALIVE) neighbours++;
            if(n_i[1] >= 0 && current_grid[n_i[1] * M + n_j[1]] == ALIVE) neighbours++;
            if(n_i[2] >= 0 && n_j[2] < M && current_grid[n_i[2] * M + n_j[2]] == ALIVE) neighbours++;
            if(n_j[3] < M && current_grid[n_i[3] * M + n_j[3]] == ALIVE) neighbours++;
            if(n_i[4] < N && n_j[4] < M && current_grid[n_i[4] * M + n_j[4]] == ALIVE) neighbours++;
            if(n_i[5] < N && current_grid[n_i[5] * M + n_j[5]] == ALIVE) neighbours++;
            if(n_i[6] < N && n_j[6] >= 0 && current_grid[n_i[6] * M + n_j[6]] == ALIVE) neighbours++;
            if(n_j[7] >= 0 && current_grid[n_i[7] * M + n_j[7]] == ALIVE) neighbours++;

            if(current_grid[i*M + j] == ALIVE && (neighbours == 2 || neighbours == 3)){
                next_grid[i*M + j] = ALIVE;
            } else if(current_grid[i*M + j] == DEAD && neighbours == 3){
                next_grid[i*M + j] = ALIVE;
            }else{
                next_grid[i*M + j] = DEAD;
            }
        }
    }
}

void game_of_life_stats(struct Options *opt, int step, int *current_grid){
    unsigned long long num_in_state[NUMSTATES];
    int m = opt->m, n = opt->n;
    for(int i = 0; i < NUMSTATES; i++) num_in_state[i] = 0;
    for(int j = 0; j < m; j++){
        for(int i = 0; i < n; i++){
            num_in_state[current_grid[i*m + j]]++;
        }
    }
    double frac, ntot = opt->m*opt->n;
    FILE *fptr;
    if (step == 0) {
        fptr = fopen(opt->statsfile, "w");
    }
    else {
        fptr = fopen(opt->statsfile, "a");
    }
    fprintf(fptr, "step %d : ", step);
    for(int i = 0; i < NUMSTATES; i++) {
        frac = (double)num_in_state[i]/ntot;
        fprintf(fptr, "Frac in state %d = %f,\t", i, frac);
    }
    fprintf(fptr, " \n");
    fclose(fptr);
}

int main(int argc, char **argv)
{
    struct timeval start_time, stop_time, elapsed_time;

    struct Options *opt = (struct Options *) malloc(sizeof(struct Options));
    opt->n = N;
    opt->m = M;
    opt->nsteps = NSTEPS;

    int *initial_grid = (int *) malloc(sizeof(int) * N * M);
    int *grid = (int *) malloc(sizeof(int) * N * M);
    int *updated_grid = (int *) malloc(sizeof(int) * N * M);
    int *cpu_final_grid = (int * )malloc(sizeof(int) * N * M);
    int *gpu_final_grid = (int * )malloc(sizeof(int) * N * M);

    int current_step = 0;
    int *tmp = NULL;
    generate_IC(opt->iictype, initial_grid, N, M);

    memcpy(grid, initial_grid, sizeof(int) * N * M);

    
    // Run GPU implementation
    gettimeofday(&start_time, NULL);
    while(current_step != NSTEPS){
        game_of_life_gpu(opt, grid, updated_grid);
        // swap current and updated grid
        tmp = grid;
        grid = updated_grid;
        updated_grid = tmp;
        current_step++;
    }
    gettimeofday(&stop_time, NULL);
    timersub(&stop_time, &start_time, &elapsed_time);
    float cpu_elapsed = elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0;
    memcpy(cpu_final_grid, updated_grid, sizeof(int) * N * M); // save result for checking

    // Run CPU implementation
    memcpy(grid, initial_grid, sizeof(int) * N * M); // fresh grid
    current_step = 0;
    gettimeofday(&start_time, NULL);
    #pragma acc enter data copyin(grid[0:N*M]), create(updated_grid[0:N*M])
    while(current_step != NSTEPS){
        game_of_life_gpu(opt, grid, updated_grid);
        // swap current and updated grid
        tmp = grid;
        grid = updated_grid;
        updated_grid = tmp;
        current_step++;
    }
    #pragma acc exit data copyout(updated_grid[0:N*M])
    #pragma acc wait
    gettimeofday(&stop_time, NULL);
    timersub(&stop_time, &start_time, &elapsed_time);
    float gpu_elapsed = elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0;
    memcpy(gpu_final_grid, updated_grid, sizeof(int) * N * M); // save result for checking

    // Check correctness
    long correct = 0;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            if(cpu_final_grid[i*M + j] == gpu_final_grid[i*M + j]) {
                correct++;
            }
        }
    }

    printf("CPU:        %f ms\n", cpu_elapsed);
    printf("GPU:        %f ms\n", gpu_elapsed);
    printf("Speedup:    %.2fx\n", cpu_elapsed / gpu_elapsed);

    // Print number of correct values (with color using escape codes)
    printf("Correct:    ");
    if (correct == N*M) { printf("\033[0;32m");} else { printf("\033[0;31m");}
    printf("%ld/%ld\n", correct, N*M);
    printf("\033[0m");

    // visualise_ascii(NSTEPS, cpu_final_grid, N, M);
    // visualise_ascii(NSTEPS, gpu_final_grid, N, M);

    FILE *fp = fopen("performance.txt", "a");
    fprintf(fp, "%d,%d,%d,%f,%f,%f\n", N, M, NSTEPS, cpu_elapsed, gpu_elapsed, cpu_elapsed / gpu_elapsed);
    fclose(fp);

    free(initial_grid);
    free(grid);
    free(updated_grid);
    free(cpu_final_grid);
    free(gpu_final_grid);
    free(opt);
    return 0;
}
