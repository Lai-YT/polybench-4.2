#include <assert.h>
#include <stdio.h>
#include "correlation_kernel.hu"
/* correlation.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>
#include <timer.hpp>
#include <texture.hu>

/* Include benchmark-specific header. */
#include "correlation.h"

#if defined(__CUDACC__) && !defined(CUDA_DEVICE)
#define CUDA_DEVICE 0
#endif

/* Array initialization. */
static
void init_array (int m,
		 int n,
		 DATA_TYPE *float_n,
		 DATA_TYPE POLYBENCH_2D(data,N,M,n,m))
{
  int i, j;

  *float_n = (DATA_TYPE)N;

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] = (DATA_TYPE)(i*j)/M + i;

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(corr,M,M,m,m))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("corr");
  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, corr[i][j]);
    }
  POLYBENCH_DUMP_END("corr");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_correlation(int m, int n,
			DATA_TYPE float_n,
			DATA_TYPE POLYBENCH_2D(data,N,M,n,m),
			DATA_TYPE POLYBENCH_2D(corr,M,M,m,m),
			DATA_TYPE POLYBENCH_1D(mean,M,m),
			DATA_TYPE POLYBENCH_1D(stddev,M,m))
{
  int i, j, k;

  DATA_TYPE eps = SCALAR_VAL(0.1);


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

    // float (*dev_corr)[1200];
    SurfaceObject_t surf_corr;
    float (*dev_data)[1200];
    float *dev_mean;
    float *dev_stddev;
    
    CpuTimer cpu_timer;
    cpu_timer.Start();
    createSurfaceObject(&surf_corr, nullptr, 1200, 1200);
    cudaCheckReturn(cudaMalloc((void **) &dev_data, (1400) * (1200) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_mean, (1200) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_stddev, (1200) * sizeof(float)));

    cudaCheckReturn(cudaMemcpy(dev_data, data, (1400) * (1200) * sizeof(float), cudaMemcpyHostToDevice));
    cpu_timer.Stop();
    printf("%lf\n", cpu_timer.ElapsedTime<TimeUnit::S>());

    GpuEventTimer gpu_timer;
    gpu_timer.Start();
    {
      dim3 k0_dimBlock(32);
      dim3 k0_dimGrid(38);
      kernel0 <<<k0_dimGrid, k0_dimBlock>>> (surf_corr.surf);
      cudaCheckKernel();
    }
    gpu_timer.Stop();
    printf("kernel0: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
    
    gpu_timer.Start();
    {
      dim3 k1_dimBlock(32);
      dim3 k1_dimGrid(38);
      kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_data, eps, float_n, dev_mean, dev_stddev);
      cudaCheckKernel();
    }
    gpu_timer.Stop();
    printf("kernel1: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
    
    gpu_timer.Start();
    {
      dim3 k2_dimBlock(16, 32);
      dim3 k2_dimGrid(38, 44);
      kernel2 <<<k2_dimGrid, k2_dimBlock>>> (dev_data, float_n, dev_mean, dev_stddev);
      cudaCheckKernel();
    }
    gpu_timer.Stop();
    printf("kernel2: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
    
    gpu_timer.Start();
    {
      dim3 k3_dimBlock(16, 32);
      dim3 k3_dimGrid(38, 38);
      kernel3 <<<k3_dimGrid, k3_dimBlock>>> (surf_corr.surf, dev_data);
      cudaCheckKernel();
    }
    gpu_timer.Stop();
    printf("kernel3: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
    
    cpu_timer.Start();
    copyFromSurfaceObject(surf_corr, corr);
    cudaCheckReturn(cudaMemcpy(data, dev_data, (1400) * (1200) * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaMemcpy(mean, dev_mean, (1200) * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaMemcpy(stddev, dev_stddev, (1200) * sizeof(float), cudaMemcpyDeviceToHost));
    corr[1199][1199] = 1.F;
    destroySurfaceObject(&surf_corr);
    cudaCheckReturn(cudaFree(dev_data));
    cudaCheckReturn(cudaFree(dev_mean));
    cudaCheckReturn(cudaFree(dev_stddev));
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
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(corr,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);

  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));

#if defined(__CUDACC__) && defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  /* Start timer. */
  polybench_start_instruments;

#if defined(__CUDACC__) && !defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  /* Run kernel. */
  kernel_correlation (m, n, float_n,
		      POLYBENCH_ARRAY(data),
		      POLYBENCH_ARRAY(corr),
		      POLYBENCH_ARRAY(mean),
		      POLYBENCH_ARRAY(stddev));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(corr)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(corr);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}
