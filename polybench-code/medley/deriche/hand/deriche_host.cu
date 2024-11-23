#include <assert.h>
#include <stdio.h>
#include "deriche_kernel.hu"
/* deriche.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>
#include <texture.hu>
#include <timer.hpp>

/* Include benchmark-specific header. */
#include "deriche.h"

#if defined(__CUDACC__) && !defined(CUDA_DEVICE)
#define CUDA_DEVICE 0
#endif

/* Array initialization. */
static
void init_array (int w, int h, DATA_TYPE* alpha,
		 DATA_TYPE POLYBENCH_2D(imgIn,W,H,w,h),
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))
{
  int i, j;

  *alpha=0.25; //parameter of the filter

  //input should be between 0 and 1 (grayscale image pixel)
  for (i = 0; i < w; i++)
     for (j = 0; j < h; j++)
	imgIn[i][j] = (DATA_TYPE) ((313*i+991*j)%65536) / 65535.0f;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int w, int h,
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("imgOut");
  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++) {
      if ((i * h + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, imgOut[i][j]);
    }
  POLYBENCH_DUMP_END("imgOut");
  POLYBENCH_DUMP_FINISH;
}



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Original code provided by Gael Deest */
static
void kernel_deriche(int w, int h, DATA_TYPE alpha,
       DATA_TYPE POLYBENCH_2D(imgIn, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y1, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y2, W, H, w, h)) {
    int i,j;
    DATA_TYPE xm1, tm1, ym1, ym2;
    DATA_TYPE xp1, xp2;
    DATA_TYPE tp1, tp2;
    DATA_TYPE yp1, yp2;

    DATA_TYPE k;
    DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
    DATA_TYPE b1, b2, c1, c2;

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

     b1 = powf(2.F, -alpha);
     k = (((1.F - expf(-alpha)) * (1.F - expf(-alpha))) / ((1.F + ((2.F * alpha) * expf(-alpha))) - expf(2.F * alpha)));
     a1 = (a5 = k);
     a2 = (a6 = ((k * expf(-alpha)) * (alpha - 1.F)));
     a3 = (a7 = ((k * expf(-alpha)) * (alpha + 1.F)));
     a4 = (a8 = ((-k) * expf((-2.F) * alpha)));
     b2 = (-expf((-2.F) * alpha));
     c1 = (c2 = 1);

     float *dev_a1;
     float *dev_a2;
     float *dev_a3;
     float *dev_a4;
     float *dev_a5;
     float *dev_a6;
     float *dev_a7;
     float *dev_a8;
     float *dev_b1;
     float *dev_b2;
     float *dev_c1;
     float *dev_c2;
     TextureObject_t tex_imgIn;
     SurfaceObject_t surf_imgOut;
     SurfaceObject_t surf_y1;
     SurfaceObject_t surf_y2;
     
     CpuTimer cpu_timer;
     cpu_timer.Start();
     /* W is actaully the height of the image and H is the width. */
     createTextureObject(&tex_imgIn, imgIn, H, W);
     createSurfaceObject(&surf_imgOut, nullptr, H, W);
     createSurfaceObject(&surf_y1, nullptr, H, W);
     createSurfaceObject(&surf_y2, nullptr, H, W);
     cudaCheckReturn(cudaMalloc((void **) &dev_a1, sizeof(float)));
     cudaCheckReturn(cudaMalloc((void **) &dev_a2, sizeof(float)));
     cudaCheckReturn(cudaMalloc((void **) &dev_a3, sizeof(float)));
     cudaCheckReturn(cudaMalloc((void **) &dev_a4, sizeof(float)));
     cudaCheckReturn(cudaMalloc((void **) &dev_a5, sizeof(float)));
     cudaCheckReturn(cudaMalloc((void **) &dev_a6, sizeof(float)));
     cudaCheckReturn(cudaMalloc((void **) &dev_a7, sizeof(float)));
     cudaCheckReturn(cudaMalloc((void **) &dev_a8, sizeof(float)));
     cudaCheckReturn(cudaMalloc((void **) &dev_b1, sizeof(float)));
     cudaCheckReturn(cudaMalloc((void **) &dev_b2, sizeof(float)));
     cudaCheckReturn(cudaMalloc((void **) &dev_c1, sizeof(float)));
     cudaCheckReturn(cudaMalloc((void **) &dev_c2, sizeof(float)));
     
     cudaCheckReturn(cudaMemcpy(dev_a1, &a1, sizeof(float), cudaMemcpyHostToDevice));
     cudaCheckReturn(cudaMemcpy(dev_a2, &a2, sizeof(float), cudaMemcpyHostToDevice));
     cudaCheckReturn(cudaMemcpy(dev_a3, &a3, sizeof(float), cudaMemcpyHostToDevice));
     cudaCheckReturn(cudaMemcpy(dev_a4, &a4, sizeof(float), cudaMemcpyHostToDevice));
     cudaCheckReturn(cudaMemcpy(dev_a5, &a5, sizeof(float), cudaMemcpyHostToDevice));
     cudaCheckReturn(cudaMemcpy(dev_a6, &a6, sizeof(float), cudaMemcpyHostToDevice));
     cudaCheckReturn(cudaMemcpy(dev_a7, &a7, sizeof(float), cudaMemcpyHostToDevice));
     cudaCheckReturn(cudaMemcpy(dev_a8, &a8, sizeof(float), cudaMemcpyHostToDevice));
     cudaCheckReturn(cudaMemcpy(dev_b1, &b1, sizeof(float), cudaMemcpyHostToDevice));
     cudaCheckReturn(cudaMemcpy(dev_b2, &b2, sizeof(float), cudaMemcpyHostToDevice));
     cudaCheckReturn(cudaMemcpy(dev_c1, &c1, sizeof(float), cudaMemcpyHostToDevice));
     cudaCheckReturn(cudaMemcpy(dev_c2, &c2, sizeof(float), cudaMemcpyHostToDevice));
     cpu_timer.Stop();
     printf("%lf\n", cpu_timer.ElapsedTime<TimeUnit::S>());

     GpuEventTimer gpu_timer;
     gpu_timer.Start();
     {
       dim3 k0_dimBlock(32);
       dim3 k0_dimGrid(128);
       kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_a3, dev_a4, dev_b1, dev_b2, tex_imgIn.tex, surf_y2.surf);
       cudaCheckKernel();
     }
     gpu_timer.Stop();
     printf("kernel0: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
     
     gpu_timer.Start();
     {
       dim3 k1_dimBlock(32);
       dim3 k1_dimGrid(128);
       kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_a1, dev_a2, dev_b1, dev_b2, tex_imgIn.tex, surf_y1.surf);
       cudaCheckKernel();
     }
     gpu_timer.Stop();
     printf("kernel1: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
     
     gpu_timer.Start();
     {
       dim3 k2_dimBlock(16, 32);
       dim3 k2_dimGrid(68, 128);
       kernel2 <<<k2_dimGrid, k2_dimBlock>>> (dev_c1, surf_imgOut.surf, surf_y1.surf, surf_y2.surf);
       cudaCheckKernel();
     }
     gpu_timer.Stop();
     printf("kernel2: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
     
     gpu_timer.Start();
     {
       dim3 k3_dimBlock(32);
       dim3 k3_dimGrid(68);
       kernel3 <<<k3_dimGrid, k3_dimBlock>>> (dev_a7, dev_a8, dev_b1, dev_b2, surf_imgOut.surf, surf_y2.surf);
       cudaCheckKernel();
     }
     gpu_timer.Stop();
     printf("kernel3: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
     
     gpu_timer.Start();
     {
       dim3 k4_dimBlock(32);
       dim3 k4_dimGrid(68);
       kernel4 <<<k4_dimGrid, k4_dimBlock>>> (dev_a5, dev_a6, dev_b1, dev_b2, surf_imgOut.surf, surf_y1.surf);
       cudaCheckKernel();
     }
     gpu_timer.Stop();
     printf("kernel4: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
     
     gpu_timer.Start();
     {
       dim3 k5_dimBlock(16, 32);
       dim3 k5_dimGrid(68, 128);
       kernel5 <<<k5_dimGrid, k5_dimBlock>>> (dev_c2, surf_imgOut.surf, surf_y1.surf, surf_y2.surf);
       cudaCheckKernel();
     }
     gpu_timer.Stop();
     printf("kernel5: %lf\n", gpu_timer.ElapsedTime<TimeUnit::S>());
     
     cpu_timer.Start();
     copyFromSurfaceObject(surf_imgOut, imgOut);
     copyFromSurfaceObject(surf_y1, y1);
     copyFromSurfaceObject(surf_y2, y2);
     cudaCheckReturn(cudaFree(dev_a1));
     cudaCheckReturn(cudaFree(dev_a2));
     cudaCheckReturn(cudaFree(dev_a3));
     cudaCheckReturn(cudaFree(dev_a4));
     cudaCheckReturn(cudaFree(dev_a5));
     cudaCheckReturn(cudaFree(dev_a6));
     cudaCheckReturn(cudaFree(dev_a7));
     cudaCheckReturn(cudaFree(dev_a8));
     cudaCheckReturn(cudaFree(dev_b1));
     cudaCheckReturn(cudaFree(dev_b2));
     cudaCheckReturn(cudaFree(dev_c1));
     cudaCheckReturn(cudaFree(dev_c2));
     destroyTextureObject(&tex_imgIn);
     destroySurfaceObject(&surf_imgOut);
     destroySurfaceObject(&surf_y1);
     destroySurfaceObject(&surf_y2);
     cpu_timer.Stop();
     printf("%lf\n", cpu_timer.ElapsedTime<TimeUnit::S>());
   }
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int w = W;
  int h = H;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(imgIn, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(imgOut, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y1, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y2, DATA_TYPE, W, H, w, h);


  /* Initialize array(s). */
  init_array (w, h, &alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut));

#if defined(__CUDACC__) && defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  /* Start timer. */
  polybench_start_instruments;

#if defined(__CUDACC__) && !defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  /* Run kernel. */
  kernel_deriche (w, h, alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut), POLYBENCH_ARRAY(y1), POLYBENCH_ARRAY(y2));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(w, h, POLYBENCH_ARRAY(imgOut)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(imgIn);
  POLYBENCH_FREE_ARRAY(imgOut);
  POLYBENCH_FREE_ARRAY(y1);
  POLYBENCH_FREE_ARRAY(y2);

  return 0;
}
