#include <assert.h>
#include <stdio.h>
#include "nussinov_kernel.hu"
/* nussinov.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>
#include <timer.hpp>
#include <texture.hu>

/* Include benchmark-specific header. */
#include "nussinov.h"

#if defined(__CUDACC__) && !defined(CUDA_DEVICE)
#define CUDA_DEVICE 0
#endif

/* RNA bases represented as chars, range is [0,3] */
// typedef char base;
// NOTE: Since PPCG failed to copy this typedef to the kernel header file, causing undefined symbol, the use of this base type is replaced by char directly.

#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

/* Array initialization. */
static
void init_array (int n,
                 char POLYBENCH_1D(seq,N,n),
		 DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j;

  //char is AGCT/0..3
  for (i=0; i <n; i++) {
     seq[i] = (char)((i+1)%4);
  }

  for (i=0; i <n; i++)
     for (j=0; j <n; j++)
       table[i][j] = 0;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(table,N,N,n,n))

{
  int i, j;
  int t = 0;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("table");
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      if (t % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, table[i][j]);
      t++;
    }
  }
  POLYBENCH_DUMP_END("table");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/*
  Original version by Dave Wonnacott at Haverford College <davew@cs.haverford.edu>,
  with help from Allison Lake, Ting Zhou, and Tian Jin,
  based on algorithm by Nussinov, described in Allison Lake's senior thesis.
*/
static
void kernel_nussinov(int n, char POLYBENCH_1D(seq,N,n),
			   DATA_TYPE POLYBENCH_2D(table,N,N,n,n))
{
  int i, j, k;

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

   char *dev_seq;
   SurfaceObject_t surf_dev_table;
   
   CpuTimer cpu_timer;
   cpu_timer.Start();
   cudaCheckReturn(cudaMalloc((void **) &dev_seq, (2500) * sizeof(char)));
   createSurfaceObject(&surf_dev_table, table, 2500, 2500);
   
   cudaCheckReturn(cudaMemcpy(dev_seq, seq, (2500) * sizeof(char), cudaMemcpyHostToDevice));
   cpu_timer.Stop();
   printf("%lf\n", cpu_timer.ElapsedTime<TimeUnit::S>());

   GpuEventTimer gpu_timer;
   gpu_timer.Start();
   for (int c0 = 1; c0 <= 2499; c0 += 1) {
     {
       dim3 k0_dimBlock(32);
       dim3 k0_dimGrid(79);
       kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_seq, surf_dev_table.surf, c0);
       cudaCheckKernel();
     }

   }
   gpu_timer.Stop();
   printf("kernel0: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
   
   cpu_timer.Start();
   copyFromSurfaceObject(surf_dev_table, table);
   cudaCheckReturn(cudaFree(dev_seq));
   destroySurfaceObject(&surf_dev_table);
   cpu_timer.Stop();
   printf("%lf\n", cpu_timer.ElapsedTime<TimeUnit::S>());
 }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(seq, char, N, n);
  POLYBENCH_2D_ARRAY_DECL(table, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

#if defined(__CUDACC__) && defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  /* Start timer. */
  polybench_start_instruments;

#if defined(__CUDACC__) && !defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  /* Run kernel. */
  kernel_nussinov (n, POLYBENCH_ARRAY(seq), POLYBENCH_ARRAY(table));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(table)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(seq);
  POLYBENCH_FREE_ARRAY(table);

  return 0;
}
