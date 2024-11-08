#include <assert.h>
#include <stdio.h>
#include "mvt_kernel.hu"
/* mvt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>
#include <timer.hpp>
#include <texture.hu>

/* Include benchmark-specific header. */
#include "mvt.h"

#if defined(__CUDACC__) && !defined(CUDA_DEVICE)
#define CUDA_DEVICE 0
#endif

/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      x1[i] = (DATA_TYPE) (i % n) / n;
      x2[i] = (DATA_TYPE) ((i + 1) % n) / n;
      y_1[i] = (DATA_TYPE) ((i + 3) % n) / n;
      y_2[i] = (DATA_TYPE) ((i + 4) % n) / n;
      for (j = 0; j < n; j++)
	A[i][j] = (DATA_TYPE) (i*j % n) / n;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x1,N,n),
		 DATA_TYPE POLYBENCH_1D(x2,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("x1");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x1[i]);
  }
  POLYBENCH_DUMP_END("x1");

  POLYBENCH_DUMP_BEGIN("x2");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, x2[i]);
  }
  POLYBENCH_DUMP_END("x2");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_mvt(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  {
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)

    TextureObject_t tex_A;
    createTextureObject(&tex_A, A, 2000, 2000);
    float *dev_x1;
    float *dev_x2;
    float *dev_y_1;
    float *dev_y_2;
    
    cudaCheckReturn(cudaMalloc((void **) &dev_x1, (2000) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_x2, (2000) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_y_1, (2000) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_y_2, (2000) * sizeof(float)));
    
    cudaCheckReturn(cudaMemcpy(dev_x1, x1, (2000) * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_x2, x2, (2000) * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_y_1, y_1, (2000) * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_y_2, y_2, (2000) * sizeof(float), cudaMemcpyHostToDevice));

    GpuEventTimer gpu_timer;
    gpu_timer.Start();
    {
      dim3 k0_dimBlock(32);
      dim3 k0_dimGrid(63);
      kernel0 <<<k0_dimGrid, k0_dimBlock>>> (tex_A.tex, dev_x1, dev_y_1);
      cudaCheckKernel();
    }
    gpu_timer.Stop();
    printf("kernel0: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
    
    gpu_timer.Start();
    {
      dim3 k1_dimBlock(32);
      dim3 k1_dimGrid(63);
      kernel1 <<<k1_dimGrid, k1_dimBlock>>> (tex_A.tex, dev_x2, dev_y_2);
      cudaCheckKernel();
    }
    gpu_timer.Stop();
    printf("kernel1: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
    
    cudaCheckReturn(cudaMemcpy(x1, dev_x1, (2000) * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaMemcpy(x2, dev_x2, (2000) * sizeof(float), cudaMemcpyDeviceToHost));
    destroyTextureObject(&tex_A);
    cudaCheckReturn(cudaFree(dev_x1));
    cudaCheckReturn(cudaFree(dev_x2));
    cudaCheckReturn(cudaFree(dev_y_1));
    cudaCheckReturn(cudaFree(dev_y_2));
  }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_1D_ARRAY_DECL(x1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(x2, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_1, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y_2, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A));

#if defined(__CUDACC__) && defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  /* Start timer. */
  polybench_start_instruments;

#if defined(__CUDACC__) && !defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  /* Run kernel. */
  kernel_mvt (n,
	      POLYBENCH_ARRAY(x1),
	      POLYBENCH_ARRAY(x2),
	      POLYBENCH_ARRAY(y_1),
	      POLYBENCH_ARRAY(y_2),
	      POLYBENCH_ARRAY(A));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x1);
  POLYBENCH_FREE_ARRAY(x2);
  POLYBENCH_FREE_ARRAY(y_1);
  POLYBENCH_FREE_ARRAY(y_2);

  return 0;
}
