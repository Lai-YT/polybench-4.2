#include <assert.h>
#include <stdio.h>
#include "adi_kernel.hu"
/* adi.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>
#include <timer.hpp>
#include <texture.hu>

/* Include benchmark-specific header. */
#include "adi.h"

#if defined(__CUDACC__) && !defined(CUDA_DEVICE)
#define CUDA_DEVICE 0
#endif

/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	u[i][j] =  (DATA_TYPE)(i + n-j) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("u");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, u[i][j]);
    }
  POLYBENCH_DUMP_END("u");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel Computers"
 * by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
static
void kernel_adi(int tsteps, int n,
		DATA_TYPE POLYBENCH_2D(u,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(v,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(p,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(q,N,N,n,n))
{
  int t, i, j;
  DATA_TYPE DX, DY, DT;
  DATA_TYPE B1, B2;
  DATA_TYPE mul1, mul2;
  DATA_TYPE a, b, c, d, e, f;


  DX = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DY = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DT = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_TSTEPS;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 /  SCALAR_VAL(2.0);
  b = SCALAR_VAL(1.0)+mul1;
  c = a;
  d = -mul2 / SCALAR_VAL(2.0);
  e = SCALAR_VAL(1.0)+mul2;
  f = d;

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

   SurfaceObject_t surf_p;
   SurfaceObject_t surf_q;
   SurfaceObject_t surf_u;
   SurfaceObject_t surf_v;
   
   CpuTimer cpu_timer;
   cpu_timer.Start();
   /* NOTE: Although the copy of p, q, and v is unnecessary, it is kept for consistency with the original code. */
   createSurfaceObject(&surf_p, p, 1000, 999);
   createSurfaceObject(&surf_q, q, 1000, 999);
   createSurfaceObject(&surf_u, u, 1000, 999);
   createSurfaceObject(&surf_v, v, 1000, 1000);
   cpu_timer.Stop();
   printf("%lf\n", cpu_timer.ElapsedTime<TimeUnit::S>());
   
   GpuEventTimer gpu_timer;
   double kernel0_time = 0, kernel1_time = 0, kernel2_time = 0, kernel3_time = 0;
   for (int c0 = 1; c0 <= 500; c0 += 1) {
     gpu_timer.Start();
     {
       dim3 k0_dimBlock(32);
       dim3 k0_dimGrid(32);
       kernel0 <<<k0_dimGrid, k0_dimBlock>>> (a, b, c, d, f, surf_p.surf, surf_q.surf, surf_u.surf, surf_v.surf, c0);
       cudaCheckKernel();
     }
     gpu_timer.Stop();
     kernel0_time += gpu_timer.ElapsedTime<TimeUnit::S>();
     
     gpu_timer.Start();
     {
       dim3 k1_dimBlock(32);
       dim3 k1_dimGrid(32);
       kernel1 <<<k1_dimGrid, k1_dimBlock>>> (surf_p.surf, surf_q.surf, surf_v.surf, c0);
       cudaCheckKernel();
     }
     gpu_timer.Stop();
     kernel1_time += gpu_timer.ElapsedTime<TimeUnit::S>();
     
     gpu_timer.Start();
     {
       dim3 k2_dimBlock(32);
       dim3 k2_dimGrid(32);
       kernel2 <<<k2_dimGrid, k2_dimBlock>>> (a, c, d, e, f, surf_p.surf, surf_q.surf, surf_u.surf, surf_v.surf, c0);
       cudaCheckKernel();
     }
     gpu_timer.Stop();
     kernel2_time += gpu_timer.ElapsedTime<TimeUnit::S>();
     
     gpu_timer.Start();
     {
       dim3 k3_dimBlock(32);
       dim3 k3_dimGrid(32);
       kernel3 <<<k3_dimGrid, k3_dimBlock>>> (surf_p.surf, surf_q.surf, surf_u.surf, c0);
       cudaCheckKernel();
     }
     gpu_timer.Stop();
     kernel3_time += gpu_timer.ElapsedTime<TimeUnit::S>();
     
   }
   printf("kernel0: %lf\n", kernel0_time);
   printf("kernel1: %lf\n", kernel1_time);
   printf("kernel2: %lf\n", kernel2_time);
   printf("kernel3: %lf\n", kernel3_time);

   cpu_timer.Start();
   copyFromSurfaceObject(surf_p, p);
   copyFromSurfaceObject(surf_q, q);
   copyFromSurfaceObject(surf_u, u);
   copyFromSurfaceObject(surf_v, v);
   destroySurfaceObject(&surf_p);
   destroySurfaceObject(&surf_q);
   destroySurfaceObject(&surf_u);
   destroySurfaceObject(&surf_v);
   cpu_timer.Stop();
   printf("%lf\n", cpu_timer.ElapsedTime<TimeUnit::S>());
 }
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(u, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(v, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(p, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(q, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(u));

#if defined(__CUDACC__) && defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  /* Start timer. */
  polybench_start_instruments;

#if defined(__CUDACC__) && !defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  /* Run kernel. */
  kernel_adi (tsteps, n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(v), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(u)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(u);
  POLYBENCH_FREE_ARRAY(v);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(q);

  return 0;
}
