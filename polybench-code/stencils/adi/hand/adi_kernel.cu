#include "adi_kernel.hu"
#include <cuda.h>
__global__ void kernel0(float a, float b, float c, float d, float f,
                        cudaSurfaceObject_t p, cudaSurfaceObject_t q,
                        cudaSurfaceObject_t u, cudaSurfaceObject_t v, int c0) {
  int b0 = blockIdx.x;
  int t0 = threadIdx.x;
  __shared__ float shared_p[32][33];
  __shared__ float shared_q[32][33];
  float private_v[1][1];

#define ppcg_min(x, y)                                                         \
  ({                                                                           \
    __typeof__(x) _x = (x);                                                    \
    __typeof__(y) _y = (y);                                                    \
    _x < _y ? _x : _y;                                                         \
  })
#define ppcg_max(x, y)                                                         \
  ({                                                                           \
    __typeof__(x) _x = (x);                                                    \
    __typeof__(y) _y = (y);                                                    \
    _x > _y ? _x : _y;                                                         \
  })
#define ppcg_fdiv_q(n, d) (((n) < 0) ? -((-(n) + (d)-1) / (d)) : (n) / (d))
  {
    for (int c2 = 0; c2 <= 998; c2 += 32) {
      for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
        for (int c4 = ppcg_max(t0, ((t0 + c2 + 31) % 32) - c2 + 1);
             c4 <= ppcg_min(32, -c2 + 1000); c4 += 32) {
          /* `c4` depends on `t0`, but is always at the x dimension. */
          // shared_p[c3][c4] = p[32 * b0 + c3 + 1][c2 + c4 - 1];
          shared_p[c3][c4] = surf2Dread<float>(p, (c2 + c4 - 1) * sizeof(float),
                                               32 * b0 + c3 + 1);
        }
      }
      for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
        for (int c4 = ppcg_max(t0, ((t0 + c2 + 31) % 32) - c2 + 1);
             c4 <= ppcg_min(32, -c2 + 1000); c4 += 32) {
          /* `c4` depends on `t0`, but is always at the x dimension. */
          // shared_q[c3][c4] = q[32 * b0 + c3 + 1][c2 + c4 - 1];
          shared_q[c3][c4] = surf2Dread<float>(q, (c2 + c4 - 1) * sizeof(float),
                                               32 * b0 + c3 + 1);
        }
      }
      __syncthreads();
      if (32 * b0 + t0 <= 997 && c2 == 0) {
        private_v[0][0] = 1.F;
        shared_p[t0][1] = 0.F;
        shared_q[t0][1] = private_v[0][0];
      }
      if (32 * b0 + t0 <= 997) {
        for (int c4 = ppcg_max(0, -c2 + 1); c4 <= ppcg_min(31, -c2 + 998);
             c4 += 1) {
          /* NOTE: Although all threads in a same warp write to the same row,
            there is no bank conflict because the width of the shared memory
            is 33. */
          shared_p[t0][c4 + 1] = ((-c) / ((a * shared_p[t0][c4]) + b));
          // shared_q[t0][c4 + 1] = ((((((-d) * u[c2 + c4][32 * b0 + t0]) +
          // ((1.F + (2.F * d)) * u[c2 + c4][32 * b0 + t0 + 1])) - (f * u[c2 +
          // c4][32 * b0 + t0 + 2])) - (a * shared_q[t0][c4])) / ((a *
          // shared_p[t0][c4]) + b));
          shared_q[t0][c4 + 1] =
              ((((((-d) * surf2Dread<float>(u, (32 * b0 + t0) * sizeof(float),
                                            c2 + c4)) +
                  ((1.F + (2.F * d)) *
                   surf2Dread<float>(u, (32 * b0 + t0 + 1) * sizeof(float),
                                     c2 + c4))) -
                 (f * surf2Dread<float>(u, (32 * b0 + t0 + 2) * sizeof(float),
                                        c2 + c4))) -
                (a * shared_q[t0][c4])) /
               ((a * shared_p[t0][c4]) + b));
        }
      }
      __syncthreads();
      if (t0 + 32 * ppcg_fdiv_q(-t0 + c2, 32) >= -30 &&
          t0 + 32 * ppcg_fdiv_q(-t0 + c2, 32) <= 967) {
        for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
          // q[32 * b0 + c3 + 1][((t0 + 31) % 32) + c2] = shared_q[c3][((t0 +
          // 31) % 32) + 1];
          surf2Dwrite(shared_q[c3][((t0 + 31) % 32) + 1], q,
                      (c2 + ((t0 + 31) % 32)) * sizeof(float),
                      32 * b0 + c3 + 1);
        }
        for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
          // p[32 * b0 + c3 + 1][((t0 + 31) % 32) + c2] = shared_p[c3][((t0 +
          // 31) % 32) + 1];
          surf2Dwrite(shared_p[c3][((t0 + 31) % 32) + 1], p,
                      (c2 + ((t0 + 31) % 32)) * sizeof(float),
                      32 * b0 + c3 + 1);
        }
      }
      __syncthreads();
    }
    if (32 * b0 + t0 <= 997) {
      // v[0][32 * b0 + t0 + 1] = private_v[0][0];
      surf2Dwrite(private_v[0][0], v, (32 * b0 + t0 + 1) * sizeof(float), 0);
    }
  }
}
__global__ void kernel1(cudaSurfaceObject_t p, cudaSurfaceObject_t q,
                        cudaSurfaceObject_t v, int c0) {
  int b0 = blockIdx.x;
  int t0 = threadIdx.x;
  //__shared__ float shared_p[32][32];
  // __shared__ float shared_q[32][32];
  __shared__ float shared_v[33][32];

#define ppcg_min(x, y)                                                         \
  ({                                                                           \
    __typeof__(x) _x = (x);                                                    \
    __typeof__(y) _y = (y);                                                    \
    _x < _y ? _x : _y;                                                         \
  })
#define ppcg_max(x, y)                                                         \
  ({                                                                           \
    __typeof__(x) _x = (x);                                                    \
    __typeof__(y) _y = (y);                                                    \
    _x > _y ? _x : _y;                                                         \
  })
  for (int c2 = 0; c2 <= 997; c2 += 32) {
    //    if (t0 + 967 >= c2) {
    //      for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
    //        shared_p[c3][t0] = p[32 * b0 + c3 + 1][t0 - c2 + 967];
    //      }
    //      for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
    //        shared_q[c3][t0] = q[32 * b0 + c3 + 1][t0 - c2 + 967];
    //      }
    //    }
    if (32 * b0 + t0 <= 998) {
      for (int c3 = ppcg_max(0, c2 - 967); c3 <= 32; c3 += 1) {
        // shared_v[c3][t0] = v[-c2 + c3 + 967][32 * b0 + t0 + 1];
        shared_v[c3][t0] = surf2Dread<float>(
            v, (32 * b0 + t0 + 1) * sizeof(float), -c2 + c3 + 967);
      }
    }
    __syncthreads();
    if (32 * b0 + t0 <= 997 && c2 == 0) {
      shared_v[32][t0] = 1.F;
    }
    if (32 * b0 + t0 <= 997) {
      for (int c4 = 0; c4 <= ppcg_min(31, -c2 + 997); c4 += 1) {
        /* FIXME: All threads in a warp read the same row from `shared_p` and
         * `shared_q`, causing bank conflicts. */
        // shared_v[-c4 + 31][t0] = ((shared_p[t0][-c4 + 31] * shared_v[-c4 +
        // 32][t0]) + shared_q[t0][-c4 + 31]);
        shared_v[-c4 + 31][t0] =
            ((surf2Dread<float>(p, (-c4 + 31 - c2 + 967) * sizeof(float),
                                32 * b0 + t0 + 1) *
              shared_v[-c4 + 32][t0]) +
             surf2Dread<float>(q, (-c4 + 31 - c2 + 967) * sizeof(float),
                               32 * b0 + t0 + 1));
      }
    }
    __syncthreads();
    if (32 * b0 + t0 <= 997) {
      for (int c3 = ppcg_max(0, c2 - 966); c3 <= 31; c3 += 1) {
        // v[-c2 + c3 + 967][32 * b0 + t0 + 1] = shared_v[c3][t0];
        surf2Dwrite<float>(shared_v[c3][t0], v,
                           (32 * b0 + t0 + 1) * sizeof(float), -c2 + c3 + 967);
      }
      if (c2 == 0) {
        // v[999][32 * b0 + t0 + 1] = shared_v[32][t0];
        surf2Dwrite<float>(shared_v[32][t0], v,
                           (32 * b0 + t0 + 1) * sizeof(float), 999);
      }
    }
    __syncthreads();
  }
}

/** Row sweep */
__global__ void kernel2(float a, float c, float d, float e, float f,
                        cudaSurfaceObject_t p, cudaSurfaceObject_t q,
                        cudaSurfaceObject_t u, cudaSurfaceObject_t v, int c0) {
  int b0 = blockIdx.x;
  int t0 = threadIdx.x;
  __shared__ float shared_p[32][33];
  __shared__ float shared_q[32][33];
  float private_u[1][1];
  //    __shared__ float shared_v[34][32];

#define ppcg_min(x, y)                                                         \
  ({                                                                           \
    __typeof__(x) _x = (x);                                                    \
    __typeof__(y) _y = (y);                                                    \
    _x < _y ? _x : _y;                                                         \
  })
#define ppcg_max(x, y)                                                         \
  ({                                                                           \
    __typeof__(x) _x = (x);                                                    \
    __typeof__(y) _y = (y);                                                    \
    _x > _y ? _x : _y;                                                         \
  })
#define ppcg_fdiv_q(n, d) (((n) < 0) ? -((-(n) + (d)-1) / (d)) : (n) / (d))
  {
    for (int c2 = 0; c2 <= 998; c2 += 32) {
      for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
        for (int c4 = ppcg_max(t0, ((t0 + c2 + 31) % 32) - c2 + 1);
             c4 <= ppcg_min(32, -c2 + 1000); c4 += 32) {
          // shared_p[c3][c4] = p[32 * b0 + c3 + 1][c2 + c4 - 1];
          shared_p[c3][c4] = surf2Dread<float>(p, (c2 + c4 - 1) * sizeof(float),
                                               32 * b0 + c3 + 1);
        }
      }
      for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
        for (int c4 = ppcg_max(t0, ((t0 + c2 + 31) % 32) - c2 + 1);
             c4 <= ppcg_min(32, -c2 + 1000); c4 += 32) {
          // shared_q[c3][c4] = q[32 * b0 + c3 + 1][c2 + c4 - 1];
          shared_q[c3][c4] = surf2Dread<float>(q, (c2 + c4 - 1) * sizeof(float),
                                               32 * b0 + c3 + 1);
        }
      }
      //      if (t0 + c2 <= 999) {
      //        for (int c3 = 0; c3 <= ppcg_min(33, -32 * b0 + 999); c3 += 1) {
      //          shared_v[c3][t0] = v[32 * b0 + c3][t0 + c2];
      //        }
      //      }
      __syncthreads();
      if (32 * b0 + t0 <= 997 && c2 == 0) {
        private_u[0][0] = 1.F;
        shared_q[t0][1] = private_u[0][0];
        shared_p[t0][1] = 0.F;
      }
      if (32 * b0 + t0 <= 997) {
        for (int c4 = ppcg_max(0, -c2 + 1); c4 <= ppcg_min(31, -c2 + 998);
             c4 += 1) {
          /* NOTE: Although all threads in a same warp write to the same row of
            `shared_p` and `shared_q`, there is no bank conflict because the
            width of the shared memory is 33. */
          shared_p[t0][c4 + 1] = ((-f) / ((d * shared_p[t0][c4]) + e));
          /* FIXME: All threads in a warp read the same row from `shared_v`,
           * causing bank conflicts. */
          // shared_q[t0][c4 + 1] = ((((((-a) * shared_v[t0][c4]) + ((1.F + (2.F
          // * a)) * shared_v[t0 + 1][c4])) - (c * shared_v[t0 + 2][c4])) - (d *
          // shared_q[t0][c4])) / ((d * shared_p[t0][c4]) + e));
          shared_q[t0][c4 + 1] =
              ((((((-a) * surf2Dread<float>(v, (c2 + c4) * sizeof(float),
                                            32 * b0 + t0)) +
                  ((1.F + (2.F * a)) *
                   surf2Dread<float>(v, (c2 + c4) * sizeof(float),
                                     32 * b0 + t0 + 1))) -
                 (c * surf2Dread<float>(v, (c2 + c4) * sizeof(float),
                                        32 * b0 + t0 + 2))) -
                (d * shared_q[t0][c4])) /
               ((d * shared_p[t0][c4]) + e));
        }
      }
      __syncthreads();
      if (t0 + 32 * ppcg_fdiv_q(-t0 + c2, 32) <= 967) {
        for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
          if (t0 == 1 && c0 == 500 && c2 == 0) {
            // q[32 * b0 + c3 + 1][0] = shared_q[c3][1];
            surf2Dwrite<float>(shared_q[c3][1], q, 0 * sizeof(float),
                               32 * b0 + c3 + 1);
          } else if (t0 + 32 * ppcg_fdiv_q(-t0 + c2, 32) >= -30) {
            // q[32 * b0 + c3 + 1][((t0 + 31) % 32) + c2] = shared_q[c3][((t0 +
            // 31) % 32) + 1];
            surf2Dwrite<float>(shared_q[c3][((t0 + 31) % 32) + 1], q,
                               (c2 + ((t0 + 31) % 32)) * sizeof(float),
                               32 * b0 + c3 + 1);
          }
        }
        for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
          if (t0 == 1 && c0 == 500 && c2 == 0) {
            // p[32 * b0 + c3 + 1][0] = shared_p[c3][1];
            surf2Dwrite<float>(shared_p[c3][1], p, 0 * sizeof(float),
                               32 * b0 + c3 + 1);
          } else if (t0 + 32 * ppcg_fdiv_q(-t0 + c2, 32) >= -30) {
            // p[32 * b0 + c3 + 1][((t0 + 31) % 32) + c2] = shared_p[c3][((t0 +
            // 31) % 32) + 1];
            surf2Dwrite<float>(shared_p[c3][((t0 + 31) % 32) + 1], p,
                               (c2 + ((t0 + 31) % 32)) * sizeof(float),
                               32 * b0 + c3 + 1);
          }
        }
      }
      __syncthreads();
    }
    if (32 * b0 + t0 <= 997) {
      /* FIXME: Uncoalesced memory access. */
      // u[32 * b0 + t0 + 1][0] = private_u[0][0];
      surf2Dwrite<float>(private_u[0][0], u, 0 * sizeof(float),
                         32 * b0 + t0 + 1);
    }
  }
}
__global__ void kernel3(cudaSurfaceObject_t p, cudaSurfaceObject_t q,
                        cudaSurfaceObject_t u, int c0) {
  int b0 = blockIdx.x;
  int t0 = threadIdx.x;
  //  __shared__ float shared_p[32][32];
  //  __shared__ float shared_q[32][32];
  __shared__ float shared_u[32][33];

#define ppcg_min(x, y)                                                         \
  ({                                                                           \
    __typeof__(x) _x = (x);                                                    \
    __typeof__(y) _y = (y);                                                    \
    _x < _y ? _x : _y;                                                         \
  })
#define ppcg_max(x, y)                                                         \
  ({                                                                           \
    __typeof__(x) _x = (x);                                                    \
    __typeof__(y) _y = (y);                                                    \
    _x > _y ? _x : _y;                                                         \
  })
#define ppcg_fdiv_q(n, d) (((n) < 0) ? -((-(n) + (d)-1) / (d)) : (n) / (d))
  for (int c2 = 0; c2 <= 997; c2 += 32) {
    //    if (t0 + 967 >= c2) {
    //      for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
    //        shared_p[c3][t0] = p[32 * b0 + c3 + 1][t0 - c2 + 967];
    //      }
    //      for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
    //        shared_q[c3][t0] = q[32 * b0 + c3 + 1][t0 - c2 + 967];
    //      }
    //    }
    for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
      for (int c4 = ppcg_max(t0, t0 + 32 * ppcg_fdiv_q(-t0 + c2 - 8, 32) - 928);
           c4 <= 32; c4 += 32) {
        // shared_u[c3][c4] = u[32 * b0 + c3 + 1][-c2 + c4 + 967];
        shared_u[c3][c4] = surf2Dread<float>(
            u, (-c2 + c4 + 967) * sizeof(float), 32 * b0 + c3 + 1);
      }
    }
    __syncthreads();
    if (32 * b0 + t0 <= 997 && c2 == 0) {
      /* NOTE: `shared_u` has width 33, so there is no bank conflict. */
      shared_u[t0][32] = 1.F;
    }
    if (32 * b0 + t0 <= 997) {
      for (int c4 = 0; c4 <= ppcg_min(31, -c2 + 997); c4 += 1) {
        /* NOTE: `shared_u` has width 33, so there is no bank conflict. */
        /* FIXME: All threads in a warp read the same row from `shared_p` and
         * `shared_q`, causing bank conflicts. */
        // shared_u[t0][-c4 + 31] = ((shared_p[t0][-c4 + 31] * shared_u[t0][-c4
        // + 32]) + shared_q[t0][-c4 + 31]);
        shared_u[t0][-c4 + 31] =
            ((surf2Dread<float>(p, (-c4 + 31 - c2 + 967) * sizeof(float),
                                32 * b0 + t0 + 1) *
              shared_u[t0][-c4 + 32]) +
             surf2Dread<float>(q, (-c4 + 31 - c2 + 967) * sizeof(float),
                               32 * b0 + t0 + 1));
      }
    }
    __syncthreads();
    if (t0 + 966 >= c2) {
      for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 997); c3 += 1) {
        // u[32 * b0 + c3 + 1][t0 - c2 + 967] = shared_u[c3][t0];
        surf2Dwrite<float>(shared_u[c3][t0], u, (t0 - c2 + 967) * sizeof(float),
                           32 * b0 + c3 + 1);
        if (t0 == 0 && c2 == 0) {
          // u[32 * b0 + c3 + 1][999] = shared_u[c3][32];
          surf2Dwrite<float>(shared_u[c3][32], u, 999 * sizeof(float),
                             32 * b0 + c3 + 1);
        }
      }
    }
    __syncthreads();
  }
}
