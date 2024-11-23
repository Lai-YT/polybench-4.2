#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
#error "This file uses APIs of surface memory, which are not supported by CUDA clang compiler."
#endif

#include "deriche_kernel.hu"

#ifdef INLINE_ASM
#include "devintrin.hu"
#endif

__global__ void kernel0(float *a3, float *a4, float *b1, float *b2, cudaTextureObject_t tex_imgIn, cudaSurfaceObject_t surf_y2)
{
    int t0 = threadIdx.x;
    int i = 32 * blockIdx.x + t0;
    __shared__ float shared_a3;
    __shared__ float shared_a4;
    __shared__ float shared_b1;
    __shared__ float shared_b2;
    float private_xp1;
    float private_xp2;
    float private_yp1;
    float private_yp2;
#ifdef NO_SURF_RAW
    // Use an additionally register to prevent read-after-write on an address of surface memory.
    float private_temp;
#endif

    {
      if (t0 == 0) {
        shared_a3 = *a3;
        shared_a4 = *a4;
        shared_b1 = *b1;
        shared_b2 = *b2;
      }
      __syncthreads();
      private_yp1 = 0.F;
      private_xp2 = 0.F;
      private_xp1 = 0.F;
      private_yp2 = 0.F;
      /* The loop is doubly negated to be incrementing. */
      /* NOTE: Due to texture throttle, you get no performance improvement from aggressive loop unrolling. */
      for (int j = -2159; j <= 0; j += 1) {
        /* FIXME: Different rows are accessed at the same time, leading to non-coalesced accesses. */
        // y2[i][-j] = ((((shared_a3 * private_xp1) + (shared_a4 * private_xp2)) + (shared_b1 * private_yp1)) + (shared_b2 * private_yp2));
#ifdef NO_SURF_RAW
        private_temp = ((((shared_a3 * private_xp1) + (shared_a4 * private_xp2)) + (shared_b1 * private_yp1)) + (shared_b2 * private_yp2));
#else
        surf2Dwrite<float>(((((shared_a3 * private_xp1) + (shared_a4 * private_xp2)) + (shared_b1 * private_yp1)) + (shared_b2 * private_yp2)), surf_y2, sizeof(float) * -j, i);
#endif
        private_xp2 = private_xp1;
        /* FIXME: Different rows are accessed at the same time, leading to non-coalesced accesses. */
        // private_xp1 = imgIn[i][-j];
#ifdef INLINE_ASM
        private_xp1 = __tex2Ds32<float>(tex_imgIn, -j, i);
#else
        private_xp1 = tex2D<float>(tex_imgIn, -j, i);
#endif
        private_yp2 = private_yp1;
        /* FIXME: Different rows are accessed at the same time, leading to non-coalesced accesses. */
        // private_yp1 = y2[i][-j];
#ifdef NO_SURF_RAW
        private_yp1 = private_temp;
        surf2Dwrite<float>(private_temp, surf_y2, sizeof(float) * -j, i);
#else
        private_yp1 = surf2Dread<float>(surf_y2, sizeof(float) * -j, i);
#endif
      }
    }
}
__global__ void kernel1(float *a1, float *a2, float *b1, float *b2, cudaTextureObject_t tex_imgIn, cudaSurfaceObject_t surf_y1)
{
    int t0 = threadIdx.x;
    int i = 32 * blockIdx.x + t0;
    __shared__ float shared_a1;
    __shared__ float shared_a2;
    __shared__ float shared_b1;
    __shared__ float shared_b2;
    float private_xm1;
    float private_ym1;
    float private_ym2;

    {
      if (t0 == 0) {
        shared_a1 = *a1;
        shared_a2 = *a2;
        shared_b1 = *b1;
        shared_b2 = *b2;
      }
      __syncthreads();
      private_ym1 = 0.F;
      private_xm1 = 0.F;
      private_ym2 = 0.F;
      for (int j = 0; j <= 2159; j += 1) {
        /* FIXME: Different rows are accessed at the same time, leading to non-coalesced accesses. */
        // y1[i][j] = ((((shared_a1 * imgIn[i][j]) + (shared_a2 * private_xm1)) + (shared_b1 * private_ym1)) + (shared_b2 * private_ym2));
#ifdef INLINE_ASM
        surf2Dwrite<float>(((((shared_a1 * __tex2Ds32<float>(tex_imgIn, j, i)) + (shared_a2 * private_xm1)) + (shared_b1 * private_ym1)) + (shared_b2 * private_ym2)), surf_y1, sizeof(float) * j, i);
#else
        surf2Dwrite<float>(((((shared_a1 * tex2D<float>(tex_imgIn, j, i)) + (shared_a2 * private_xm1)) + (shared_b1 * private_ym1)) + (shared_b2 * private_ym2)), surf_y1, sizeof(float) * j, i);
#endif
        /* FIXME: Different rows are accessed at the same time, leading to non-coalesced accesses. */
        // private_xm1 = imgIn[i][j];
#ifdef INLINE_ASM
        private_xm1 = __tex2Ds32<float>(tex_imgIn, j, i);
#else
        private_xm1 = tex2D<float>(tex_imgIn, j, i);
#endif
        private_ym2 = private_ym1;
        /* FIXME: Different rows are accessed at the same time, leading to non-coalesced accesses. */
        // private_ym1 = y1[i][j];
        private_ym1 = surf2Dread<float>(surf_y1, sizeof(float) * j, i);
      }
    }
}
__global__ void kernel2(float *c1, cudaSurfaceObject_t surf_imgOut, cudaSurfaceObject_t surf_y1, cudaSurfaceObject_t surf_y2)
{
    int t0 = threadIdx.y, t1 = threadIdx.x;
    int i = 32 * blockIdx.y + t0;
    int j = 32 * blockIdx.x;
    __shared__ float shared_c1;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (t0 == 0 && t1 == 0) {
        shared_c1 = *c1;
      }
      __syncthreads();
#ifdef LOOP_VERSIONING
      if (2159 - j <= 31) {
      for (int jj = t1; jj <= 2159 - j; jj += 16)
#else
      for (int jj = t1; jj <= ppcg_min(31, 2159 - j); jj += 16)
#endif
      {
        /* For each threads in a warp, the `i` is the same, `jj` is consecutive.
         * So the accesses are coalesced.
         */
        /* NOTE: Since we're using surface memory, all uses are affected. */
        // imgOut[i][j + jj] = (shared_c1 * (y1[i][j + jj] + y2[i][j + jj]));
        surf2Dwrite<float>((shared_c1 * (surf2Dread<float>(surf_y1, sizeof(float) * (j + jj), i) + surf2Dread<float>(surf_y2, sizeof(float) * (j + jj), i))), surf_imgOut, sizeof(float) * (j + jj), i);
      }
#ifdef LOOP_VERSIONING
      } else {
      for (int jj = t1; jj <= 31; jj += 16) {
        surf2Dwrite<float>((shared_c1 * (surf2Dread<float>(surf_y1, sizeof(float) * (j + jj), i) + surf2Dread<float>(surf_y2, sizeof(float) * (j + jj), i))), surf_imgOut, sizeof(float) * (j + jj), i);
      }
      }
#endif
    }
}
__global__ void kernel3(float *a7, float *a8, float *b1, float *b2, cudaSurfaceObject_t surf_imgOut, cudaSurfaceObject_t surf_y2)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int j = 32 * b0 + t0;
    __shared__ float shared_a7;
    __shared__ float shared_a8;
    __shared__ float shared_b1;
    __shared__ float shared_b2;
    float private_tp1;
    float private_tp2;
    float private_yp1;
    float private_yp2;

    {
      if (t0 == 0) {
        shared_a7 = *a7;
        shared_a8 = *a8;
        shared_b1 = *b1;
        shared_b2 = *b2;
      }
      __syncthreads();
      if (j <= 2159) {
        private_yp1 = 0.F;
        private_yp2 = 0.F;
        private_tp1 = 0.F;
        private_tp2 = 0.F;
        for (int i = -4095; i <= 0; i += 1) {
          /* `i` is the same, `j` is consecutive; well coalesced. */
          /* NOTE: Since we're using surface memory, all uses are affected. */
          // y2[-i][j] = ((((shared_a7 * private_tp1) + (shared_a8 * private_tp2)) + (shared_b1 * private_yp1)) + (shared_b2 * private_yp2));
          surf2Dwrite<float>(((((shared_a7 * private_tp1) + (shared_a8 * private_tp2)) + (shared_b1 * private_yp1)) + (shared_b2 * private_yp2)), surf_y2, sizeof(float) * j, -i);
          private_tp2 = private_tp1;
          /* `i` is the same, `j` is consecutive; well coalesced. */
          // private_tp1 = imgOut[-i][j];
          private_tp1 = surf2Dread<float>(surf_imgOut, sizeof(float) * j, -i);
          private_yp2 = private_yp1;
          /* `i` is the same, `j` is consecutive; well coalesced. */
          // private_yp1 = y2[-i][j];
          private_yp1 = surf2Dread<float>(surf_y2, sizeof(float) * j, -i);
        }
      }
    }
}
__global__ void kernel4(float *a5, float *a6, float *b1, float *b2, cudaSurfaceObject_t surf_imgOut, cudaSurfaceObject_t surf_y1)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int j = 32 * b0 + t0;
    __shared__ float shared_a5;
    __shared__ float shared_a6;
    __shared__ float shared_b1;
    __shared__ float shared_b2;
    float private_tm1;
    float private_ym1;
    float private_ym2;

    {
      if (t0 == 0) {
        shared_a5 = *a5;
        shared_a6 = *a6;
        shared_b1 = *b1;
        shared_b2 = *b2;
      }
      __syncthreads();
      if (j <= 2159) {
        private_ym1 = 0.F;
        private_tm1 = 0.F;
        private_ym2 = 0.F;
        for (int i = 0; i <= 4095; i += 1) {
          /* `i` is the same, `j` is consecutive; well coalesced. */
          /* NOTE: Since we're using surface memory, all uses are affected. */
          // y1[i][j] = ((((shared_a5 * imgOut[i][j]) + (shared_a6 * private_tm1)) + (shared_b1 * private_ym1)) + (shared_b2 * private_ym2));
          surf2Dwrite<float>(((((shared_a5 * surf2Dread<float>(surf_imgOut, sizeof(float) * j, i)) + (shared_a6 * private_tm1)) + (shared_b1 * private_ym1)) + (shared_b2 * private_ym2)), surf_y1, sizeof(float) * j, i);
          private_ym2 = private_ym1;
          /* `i` is the same, `j` is consecutive; well coalesced. */
          // private_ym1 = y1[i][j];
          private_ym1 = surf2Dread<float>(surf_y1, sizeof(float) * j, i);
          // private_tm1 = imgOut[i][j];
          private_tm1 = surf2Dread<float>(surf_imgOut, sizeof(float) * j, i);
        }
      }
    }
}
__global__ void kernel5(float *c2, cudaSurfaceObject_t surf_imgOut, cudaSurfaceObject_t surf_y1, cudaSurfaceObject_t surf_y2)
{
    int t0 = threadIdx.y, t1 = threadIdx.x;
    int i = 32 * blockIdx.y + t0;
    int j = 32 * blockIdx.x;
    __shared__ float shared_c2;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (t0 == 0 && t1 == 0) {
        shared_c2 = *c2;
      }
      __syncthreads();
#ifdef LOOP_VERSIONING
      if (2159 - j <= 31) {
      for (int jj = t1; jj <= 2159 - j; jj += 16)
#else
      for (int jj = t1; jj <= ppcg_min(31, 2159 - j); jj += 16)
#endif
      {
        /* For each threads in a warp, the `i` is the same, `jj` is consecutive.
         * So the accesses are coalesced.
         */
        /* NOTE: Since we're using surface memory, all uses are affected. */
        // imgOut[i][j + jj] = (shared_c2 * (y1[i][j + jj] + y2[i][j + jj]));
        surf2Dwrite<float>((shared_c2 * (surf2Dread<float>(surf_y1, sizeof(float) * (j + jj), i) + surf2Dread<float>(surf_y2, sizeof(float) * (j + jj), i))), surf_imgOut, sizeof(float) * (j + jj), i);
      }
#ifdef LOOP_VERSIONING
      } else {
      for (int jj = t1; jj <= 31; jj += 16) {
        surf2Dwrite<float>((shared_c2 * (surf2Dread<float>(surf_y1, sizeof(float) * (j + jj), i) + surf2Dread<float>(surf_y2, sizeof(float) * (j + jj), i))), surf_imgOut, sizeof(float) * (j + jj), i);
      }
      }
#endif
    }
}
