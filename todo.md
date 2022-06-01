# Game-of-Life GPU assignment


## Exercise 1 - CUDA
### Notes

### Environment
```
module load gcc cuda
```

### Checklist
- [x] Compile and run serial version
- [x] Port GOL to GPU manually using CUDA
- [x] Check correctness
- [x] Performance evaluation
- [ ] Run through memory debugger/profiler to check for leaks


## Exercise 2 - OpenACC
### Notes
- n_i[] and n_j[] should be kept private
- T[][] should be shared
- 
### Environment
```
module load pgi/19.7
```

### Checklist
- [x] Compile and run serial version
- [x] Profile code to find regions of interest
- [x] Implement appropriate parallelisms
- [x] Handle data transfers correctly

## Additional
- [ ] Test for when GPU offload is worth it
- [ ] Trap for size to determine whether to GPU offload or not

### Investigate
```
#pragma acc enter data
#pragma acc exit data
```
- CUDA C programming guide
- OpenACC Reference Guide

