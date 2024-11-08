#include "mvt_kernel.hu"
#include <devintrin.hu>

/**
 * x1 += A * y1
 */
__global__ void kernel0(cudaTextureObject_t tex_A, float x1[2000], float y_1[2000])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int x = 32 * b0 + t0;
    float private_x1[1];
    __shared__ float shared_y_1[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (x <= 1999) {
        private_x1[0] = x1[x];
      }
      /* Strip mining of size 32. */
      for (int c1 = 0; c1 <= 1999; c1 += 32) {
        if (t0 + c1 <= 1999) {
          /* Well coalesced. */
          shared_y_1[t0] = y_1[t0 + c1];
        }
        __syncthreads();
        if (x <= 1999) {
#ifdef LOOP_VERSIONING
        if (1999 - c1 <= 31) {
          for (int c3 = 0; c3 <= 1999 - c1; c3 += 1)
#else
          for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 1999); c3 += 1)
#endif
          {
            /* FIXME: 32-way bank conflict on access to shared array shared_A. */
            // private_x1[0] = (private_x1[0] + (shared_A[t0][c3] * shared_y_1[c3]));
#ifdef INLINE_ASM
            private_x1[0] = (private_x1[0] + (__tex2Ds32<float>(tex_A, c1 + c3, x) * shared_y_1[c3]));
#else
            private_x1[0] = (private_x1[0] + (tex2D<float>(tex_A, c1 + c3, x) * shared_y_1[c3]));
#endif
          }
#ifdef LOOP_VERSIONING
        } else {
          for (int c3 = 0; c3 <= 31; c3 += 1) {
#ifdef INLINE_ASM
            private_x1[0] = (private_x1[0] + (__tex2Ds32<float>(tex_A, c1 + c3, x) * shared_y_1[c3]));
#else
            private_x1[0] = (private_x1[0] + (tex2D<float>(tex_A, c1 + c3, x) * shared_y_1[c3]));
#endif
          }
        }
#endif
        }
        __syncthreads();
      }
      if (x <= 1999) {
        x1[x] = private_x1[0];
      }
    }
}

/**
 * x2 += A' * y2
 */
__global__ void kernel1(cudaTextureObject_t tex_A, float x2[2000], float y_2[2000])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int x = 32 * b0 + t0;
    float private_x2[1];
    __shared__ float shared_y_2[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (x <= 1999) {
        private_x2[0] = x2[x];
      }
      for (int c1 = 0; c1 <= 1999; c1 += 32) {
        if (t0 + c1 <= 1999) {
          shared_y_2[t0] = y_2[t0 + c1];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 1999) {
#ifdef LOOP_VERSIONING
        if (1999 - c1 <= 31) {
          for (int c3 = 0; c3 <= 1999 - c1; c3 += 1)
#else
          for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 1999); c3 += 1)
#endif
          {
            /* NOTE: Since we access A in a transposed way, it is well coalesced. */
            /* NOTE: Due to the use of global memory, we no longer access global memory directly. */
            // private_x2[0] = (private_x2[0] + (A[c1 + c3][x] * shared_y_2[c3]));
#ifdef INLINE_ASM
            private_x2[0] = (private_x2[0] + (__tex2Ds32<float>(tex_A, x, c1 + c3) * shared_y_2[c3]));
#else 
            private_x2[0] = (private_x2[0] + (tex2D<float>(tex_A, x, c1 + c3) * shared_y_2[c3]));
#endif
          }
#ifdef LOOP_VERSIONING
        } else {
          for (int c3 = 0; c3 <= 31; c3 += 1) {
#ifdef INLINE_ASM
            private_x2[0] = (private_x2[0] + (__tex2Ds32<float>(tex_A, x, c1 + c3) * shared_y_2[c3]));
#else
            private_x2[0] = (private_x2[0] + (tex2D<float>(tex_A, x, c1 + c3) * shared_y_2[c3]));
#endif
          }
        }
#endif
        }
        __syncthreads();
      }
      if (x <= 1999) {
        x2[x] = private_x2[0];
      }
    }
}
