.PHONY : dirs

COMMONFLAGS=-O0 -g #-DUSEPNG
INCLUDES=#-I/usr/include
LIBS=#-L/usr/lib64/ -lpng

all : dirs bin/game_of_life bin/game_of_life_cuda

clean :
	rm obj/*
	rm bin/*

# ============== BEGIN CUDA C++ =======================================

obj/common.o : src/common.h src/common.c
	nvcc $(COMMONFLAGS) $(INCLUDES) -o $@ -c src/common.c $(LIBS)

bin/game_of_life : src/game_of_life.c obj/common.o
	nvcc $(COMMONFLAGS) $(INCLUDES) -o $@ src/game_of_life.c obj/common.o $(LIBS)

bin/game_of_life_cuda : src/game_of_life_cuda.cu obj/common.o
	nvcc -G -lineinfo $(COMMONFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# ============== END CUDA C++ ===========================================
 