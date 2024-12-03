#include <cuda.h>
#include "correlation_kernel.hu"

__global__ void kernel0(cudaSurfaceObject_t corr)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int x = 32 * b0 + t0;

    if (x <= 1198) {
      /* FIXME: uncoalesced accesses. */
      // corr[x][x] = 1.F;
      surf2Dwrite<float>(1.F, corr, x * sizeof(float), x);
    }
}
__global__ void kernel1(float data[1400][1200], float eps, float float_n, float mean[1200], float stddev[1200])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    float private_mean[1];
    float private_stddev[1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + t0 <= 1199) {
      private_mean[0] = 0.F;
      for (int c1 = 0; c1 <= 1399; c1 += 32) {
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 1399); c3 += 1) {
          private_mean[0] += data[c1 + c3][32 * b0 + t0];
        }
      }
      private_mean[0] /= float_n;
      private_stddev[0] = 0.F;
      for (int c1 = 1376; c1 <= 2798; c1 += 32) {
        for (int c3 = ppcg_max(0, -c1 + 1399); c3 <= ppcg_min(31, -c1 + 2798); c3 += 1) {
          private_stddev[0] += ((data[c1 + c3 - 1399][32 * b0 + t0] - private_mean[0]) * (data[c1 + c3 - 1399][32 * b0 + t0] - private_mean[0]));
        }
      }
      private_stddev[0] /= float_n;
      private_stddev[0] = sqrtf(private_stddev[0]);
      private_stddev[0] = ((private_stddev[0] <= eps) ? 1.F : private_stddev[0]);
      stddev[32 * b0 + t0] = private_stddev[0];
      mean[32 * b0 + t0] = private_mean[0];
    }
}
__global__ void kernel2(float data[1400][1200], float float_n, float mean[1200], float stddev[1200])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    float private_data[1][2];
    __shared__ float shared_mean[32];
    __shared__ float shared_stddev[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (32 * b0 + t0 <= 1399) {
        private_data[0][0] = data[32 * b0 + t0][32 * b1 + t1];
        if (32 * b1 + t1 <= 1183) {
          private_data[0][1] = data[32 * b0 + t0][32 * b1 + t1 + 16];
        }
        if (t0 == 0) {
          for (int c0 = t1; c0 <= ppcg_min(31, -32 * b1 + 1199); c0 += 16) {
            shared_mean[c0] = mean[32 * b1 + c0];
          }
          for (int c0 = t1; c0 <= ppcg_min(31, -32 * b1 + 1199); c0 += 16) {
            shared_stddev[c0] = stddev[32 * b1 + c0];
          }
        }
      }
      __syncthreads();
      if (32 * b0 + t0 <= 1399) {
        private_data[0][0] -= shared_mean[t1];
        if (32 * b1 + t1 <= 1183) {
          private_data[0][1] -= shared_mean[t1 + 16];
        }
        private_data[0][0] /= (sqrtf(float_n) * shared_stddev[t1]);
        if (32 * b1 + t1 <= 1183) {
          private_data[0][1] /= (sqrtf(float_n) * shared_stddev[t1 + 16]);
        }
        data[32 * b0 + t0][32 * b1 + t1] = private_data[0][0];
        if (32 * b1 + t1 <= 1183) {
          data[32 * b0 + t0][32 * b1 + t1 + 16] = private_data[0][1];
        }
      }
    }
}
__global__ void kernel3(cudaSurfaceObject_t corr, float data[1400][1200])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    int y = 32 * b0 + t0;
    int x = 32 * b1 + t1;
    float private_corr_0[1][2];
    __shared__ float shared_data_0[32][32];
    __shared__ float shared_data_1[32][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (b1 >= b0) {
        for (int c2 = 0; c2 <= 1399; c2 += 32) {
          if (t0 + c2 <= 1399) {
            for (int c4 = t1; c4 <= ppcg_min(31, -32 * b0 + 1199); c4 += 16) {
              shared_data_0[t0][c4] = data[t0 + c2][32 * b0 + c4];
            }
            for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 1198); c4 += 16) {
              shared_data_1[t0][c4] = data[t0 + c2][32 * b1 + c4 + 1];
            }
          }
          __syncthreads();
          if (y <= 1198 && x + 16 >= y && x <= 1198 && c2 == 0) {
            if (x >= y) {
              private_corr_0[0][0] = 0.F;
            }
            if (x <= 1182) {
              private_corr_0[0][1] = 0.F;
            }
          }
          if (y <= 1198 && x + 16 >= y && x <= 1198) {
            for (int c3 = 0; c3 <= ppcg_min(31, -c2 + 1399); c3 += 1) {
              if (x >= y) {
                private_corr_0[0][0] += (shared_data_0[c3][t0] * shared_data_1[c3][t1]);
              }
              if (x <= 1182) {
                private_corr_0[0][1] += (shared_data_0[c3][t0] * shared_data_1[c3][t1 + 16]);
              }
            }
            if (c2 == 1376) {
              if (x >= y) {
                /* FIXME: shared memory bank conflict. */
                // shared_corr_1[t1][t0] = private_corr_0[0][0];
                surf2Dwrite<float>(private_corr_0[0][0], corr, y * sizeof(float), x + 1);
              }
              if (x <= 1182) {
                /* FIXME: shared memory bank conflict. */
                // shared_corr_1[t1 + 16][t0] = private_corr_0[0][1];
                surf2Dwrite<float>(private_corr_0[0][1], corr, y * sizeof(float), x + 17);
              }
            }
          }
          __syncthreads();
        }
      }
//    __syncthreads();
//    if (32 * b1 + t0 <= 1198) {
//      for (int c1 = t1; c1 <= ppcg_min(31, -32 * b0 + 32 * b1 + t0); c1 += 16) {
//        corr[32 * b1 + t0 + 1][32 * b0 + c1] = shared_corr_1[t0][c1];
//      }
//    }
      if (y <= 1198 && x + 16 >= y && x <= 1198) {
        if (x >= y) {
          // corr[y][x + 1] = private_corr_0[0][0];
          surf2Dwrite<float>(private_corr_0[0][0], corr, (x + 1) * sizeof(float), y);
        }
        if (x <= 1182) {
          // corr[y][x + 17] = private_corr_0[0][1];
          surf2Dwrite<float>(private_corr_0[0][1], corr, (x + 17) * sizeof(float), y);
        }
      }
    }
}
