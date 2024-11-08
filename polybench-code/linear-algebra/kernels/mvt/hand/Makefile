ARCH = 86
NVCC = clang++-18
UTILS = ../../../../utilities
OPT = 3
NVCCFLAGS = -I $(UTILS) -DPOLYBENCH_TIME -DPOLYBENCH_TIME_NO_CUDA_INIT_CTX -g -DDATA_TYPE_IS_FLOAT -DPOLYBENCH_USE_SCALAR_LB -DPOLYBENCH_DUMP_ARRAYS -O$(OPT)
LDFLAGS =

EXEC = mvt
ifeq ($(findstring nvcc, $(NVCC)), nvcc)
	# Since optimized debugging not supported, we cannot add the flag -G.
	NVCCFLAGS += -arch=sm_$(ARCH) -diag-suppress 550,177 -Xptxas -O$(OPT)
	EXEC := $(EXEC)-nvcc
else ifeq ($(findstring clang++, $(NVCC)), clang++)
	NVCCFLAGS += --cuda-gpu-arch=sm_$(ARCH) -Wno-unknown-cuda-version
	LDFLAGS += -lcudart -lrt -ldl -L/usr/local/cuda/lib64
endif

ifeq ($(loop-versioning), 1)
	NVCCFLAGS += -DLOOP_VERSIONING
	EXEC := $(EXEC)-loop-versioning
endif
ifeq ($(inline-asm), 1)
	NVCCFLAGS += -DINLINE_ASM
	EXEC := $(EXEC)-inline-asm
endif

$(EXEC): mvt_host.cu mvt_kernel.cu polybench.o $(UTILS)/polybench.h $(UTILS)/texture.hu $(UTILS)/timer.hpp
	$(NVCC) -o $@ $(NVCCFLAGS) $(LDFLAGS) mvt_host.cu mvt_kernel.cu polybench.o

polybench.o: $(UTILS)/polybench.c $(UTILS)/polybench.h
	$(NVCC) -o $@ -x cu -c $(UTILS)/polybench.c $(NVCCFLAGS)

clean:
	$(RM) polybench.o
	find . -type f -executable -delete
.PHONY: clean

help:
	@echo "Usage: make [FLAG=1]..."
	@echo ""
	@echo "FLAG:"
	@echo "  loop-versioning"
	@echo "  inline-asm"
.PHONY: help
