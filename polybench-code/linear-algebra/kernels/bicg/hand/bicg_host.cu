#include <assert.h>
#include <stdio.h>
#include "bicg_kernel.hu"
/* bicg.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

/* Include polybench common header. */
#include <polybench.h>
#include <texture.hu>
#include <timer.hpp>

/* Include benchmark-specific header. */
#include "bicg.h"

#if defined(__CUDACC__) && !defined(CUDA_DEVICE)
#define CUDA_DEVICE 0
#endif

/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m))
{
  int i, j;

  for (i = 0; i < m; i++)
    p[i] = (DATA_TYPE)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (DATA_TYPE)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % n)/n;
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("s");
  for (i = 0; i < m; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, s[i]);
  }
  POLYBENCH_DUMP_END("s");
  POLYBENCH_DUMP_BEGIN("q");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, q[i]);
  }
  POLYBENCH_DUMP_END("q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_bicg(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n))
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

    CpuTimer cpu_timer;
    cpu_timer.Start();
  // Duplicate A for kernel1, so that it doesn't have to use texture memory.
#ifdef DEVICE_TO_DEVICE_COPY
    float (*dev_A)[1900];
#endif
    TextureObject_t tex_A;
    createTextureObject(&tex_A, A, M, N);
    float *dev_p;
    float *dev_q;
    float *dev_r;
    float *dev_s;
    
#ifdef DEVICE_TO_DEVICE_COPY
    cudaCheckReturn(cudaMalloc((void **) &dev_A, (2100) * (1900) * sizeof(float)));
#endif
    cudaCheckReturn(cudaMalloc((void **) &dev_p, (1900) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_q, (2100) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_r, (2100) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_s, (1900) * sizeof(float))); 

#ifdef DEVICE_TO_DEVICE_COPY
    cudaCheckReturn(cudaMemcpy2DFromArray(dev_A, 1900 * sizeof(float), tex_A.devPtr,
					  0, 0, 1900 * sizeof(float), 2100,
					  cudaMemcpyDeviceToDevice));
#endif
    cudaCheckReturn(cudaMemcpy(dev_p, p, (1900) * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_r, r, (2100) * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaDeviceSynchronize());
    cpu_timer.Stop();
    printf("%lf\n", cpu_timer.ElapsedTime<TimeUnit::S>());

    GpuEventTimer gpu_timer;

    gpu_timer.Start();
    {
      dim3 k0_dimBlock(32);
      dim3 k0_dimGrid(66);
      kernel0 <<<k0_dimGrid, k0_dimBlock>>> (tex_A.tex, dev_p, dev_q);
      cudaCheckKernel();
    }
    gpu_timer.Stop();
    printf("kernel0: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
    
    gpu_timer.Start();
    {
      dim3 k1_dimBlock(32);
      dim3 k1_dimGrid(60);
#ifdef DEVICE_TO_DEVICE_COPY
      kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_A, dev_r, dev_s);
#else
      kernel1 <<<k1_dimGrid, k1_dimBlock>>> (tex_A.tex, dev_r, dev_s);
#endif
      cudaCheckKernel();
    }
    gpu_timer.Stop();
    printf("kernel1: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
    
    cpu_timer.Start();
    cudaCheckReturn(cudaMemcpy(q, dev_q, (2100) * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaMemcpy(s, dev_s, (1900) * sizeof(float), cudaMemcpyDeviceToHost));
#ifdef DEVICE_TO_DEVICE_COPY
    cudaCheckReturn(cudaFree(dev_A));
#endif
    destroyTextureObject(&tex_A);
    cudaCheckReturn(cudaFree(dev_p));
    cudaCheckReturn(cudaFree(dev_q));
    cudaCheckReturn(cudaFree(dev_r));
    cudaCheckReturn(cudaFree(dev_s));
    cpu_timer.Stop();
    printf("%lf\n", cpu_timer.ElapsedTime<TimeUnit::S>());
  }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(r),
	      POLYBENCH_ARRAY(p));

#if defined(__CUDACC__) && defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  /* Start timer. */
  polybench_start_instruments;

#if defined(__CUDACC__) && !defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  /* Run kernel. */
  kernel_bicg (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(s),
	       POLYBENCH_ARRAY(q),
	       POLYBENCH_ARRAY(p),
	       POLYBENCH_ARRAY(r));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(s);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(r);

  return 0;
}
