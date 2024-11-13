#include "bicg_kernel.hu"
#include <devintrin.hu>

/**
 * q = A * p
 */
__global__ void kernel0(cudaTextureObject_t tex_A, float p[1900], float q[2100])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int x = 32 * b0 + t0;
    __shared__ float shared_p[32];
    float private_q[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
#ifdef NOUNROLL
#pragma nounroll
#endif
      /* Strip-mined with a strip size of 32. */
      for (int c1 = 0; c1 <= 1899; c1 += 32) {
        if (t0 + c1 <= 1899) {
//        for (int c2 = 0; c2 <= ppcg_min(31, -32 * b0 + 2099); c2 += 1) {
//          /* A warp loads a row together at a time; well coalesced. */
//          shared_A[c2][t0] = A[32 * b0 + c2][t0 + c1];
//        }
          shared_p[t0] = p[t0 + c1];
        }
        __syncthreads();
        if (x <= 2099 && c1 == 0) {
          private_q[0] = 0.F;
        }
        if (x <= 2099) {
#ifdef LOOP_VERSIONING
        if (1899 - c1 <= 31) {
#ifdef NOUNROLL
#pragma nounroll
#endif
          for (int c3 = 0; c3 <= -c1 + 1899; c3 += 1)
#else
#ifdef NOUNROLL
#pragma nounroll
#endif
          for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 1899); c3 += 1)
#endif
          {
            /* FIXME: A column of shared_A is read at a time,
             * causing bank conflicts.
             */
            /* All threads in a warp read the same value of shared_p. */
            // private_q[0] = (private_q[0] + (shared_A[t0][c3] * shared_p[c3]));
#ifdef INLINE_ASM
            private_q[0] = (private_q[0] + (__tex2Ds32<float>(tex_A, c1 + c3, x) * shared_p[c3]));
#else
            private_q[0] = (private_q[0] + (tex2D<float>(tex_A, c1 + c3, x) * shared_p[c3]));
#endif
          }
#ifdef LOOP_VERSIONING
        } else {
#ifdef NOUNROLL
#pragma nounroll
#endif
          for (int c3 = 0; c3 <= 31; c3 += 1) {
#ifdef INLINE_ASM
            private_q[0] = (private_q[0] + (__tex2Ds32<float>(tex_A, c1 + c3, x) * shared_p[c3]));
#else
            private_q[0] = (private_q[0] + (tex2D<float>(tex_A, c1 + c3, x) * shared_p[c3]));
#endif
          }
        }
#endif
        }
        __syncthreads();
      }
      if (x <= 2099) {
        q[x] = private_q[0];
      }
    }
}

/**
 * s = transpose(A) * r
 */ 
__global__ void kernel1(
#ifdef DEVICE_TO_DEVICE_COPY
  float A[2100][1900],
#else
  cudaTextureObject_t tex_A,
#endif
  float r[2100], float s[1900])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int x = 32 * b0 + t0;
    __shared__ float shared_r[32];
    float private_s[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      for (int c1 = 0; c1 <= 2099; c1 += 32) {
        if (t0 + c1 <= 2099) {
          /* Well coalesced. */
          shared_r[t0] = r[t0 + c1];
        }
        __syncthreads();
        if (x <= 1899 && c1 == 0) {
          private_s[0] = 0;
        }
        if (x <= 1899) {
/* NOTE: Global memory access is known to be slower with loop versioning. */
#if defined(LOOP_VERSIONING) && ! defined(DEVICE_TO_DEVICE_COPY)
        if (2099 - c1 <= 31) {
#ifdef NOUNROLL
#pragma nounroll
#endif
          for (int c3 = 0; c3 <= -c1 + 2099; c3 += 1)
#else
#ifdef NOUNROLL
#pragma nounroll
#endif
          for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 2099); c3 += 1)
#endif
          {
#ifdef DEVICE_TO_DEVICE_COPY
            /* A row of A is read at a time; well coalesced. */
            /* NOTE: Due to the use of global memory, we no longer access global memory directly. */
            private_s[0] = (private_s[0] + (shared_r[c3] * A[c1 + c3][x]));
#else
#ifdef INLINE_ASM
            private_s[0] = (private_s[0] + (shared_r[c3] * __tex2Ds32<float>(tex_A, x, c1 + c3)));
#else
            private_s[0] = (private_s[0] + (shared_r[c3] * tex2D<float>(tex_A, x, c1 + c3)));
#endif
#endif
          }
#if defined(LOOP_VERSIONING) && ! defined(DEVICE_TO_DEVICE_COPY)
        } else {
#ifdef NOUNROLL
#pragma nounroll
#endif
          for (int c3 = 0; c3 <= 31; c3 += 1) {
#ifdef INLINE_ASM
            private_s[0] = (private_s[0] + (shared_r[c3] * __tex2Ds32<float>(tex_A, x, c1 + c3)));
#else
            private_s[0] = (private_s[0] + (shared_r[c3] * tex2D<float>(tex_A, x, c1 + c3)));
#endif
          }
        }
#endif
        }
        __syncthreads();
      }
      if (x <= 1899) {
        s[x] = private_s[0];
      }
    }
}
