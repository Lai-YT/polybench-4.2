#include "nussinov_kernel.hu"
__global__ void kernel0(char seq[2500], cudaSurfaceObject_t table, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int x = 32 * b0 + t0;
    float private_table_0[1][1];
    __shared__ float shared_table_1[63][32];
    __shared__ float shared_table_5[32][63];
    float private_table_6[1][1];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + c0 <= 2499) {
      if (x + c0 <= 2499) {
        /* FIXME: Threads in a warp are accessing different rows of table, causing uncoalesced memory access. */
        // private_table_0[0][0] = table[x][x + c0];
        private_table_0[0][0] = surf2Dread<float>(table, (x + c0) * sizeof(float), x);
        if (c0 == 1) {
          // private_table_6[0][0] = table[x + 1][x];
          private_table_6[0][0] = surf2Dread<float>(table, (x + 1) * sizeof(float), x);
        }
      }
      if (c0 >= 2) {
        for (int c2 = 0; c2 < c0; c2 += 32) {
          if (x + c0 <= 2499) {
#ifdef LOOP_VERSIONING
          if (-32 * b0 - c2 + 2498 <= 62) {
            for (int c3 = 0; c3 <= -32 * b0 -c2 + 2498; c3 += 1)
#else
            for (int c3 = 0; c3 <= ppcg_min(62, -32 * b0 - c2 + 2498); c3 += 1)
#endif
            {
              /* The only variant is t0 (x), which only appears in the x index of table. */
              // shared_table_1[c3][t0] = table[32 * b0 + c2 + c3 + 1][x + c0];
              shared_table_1[c3][t0] = surf2Dread<float>(table, (x + c0) * sizeof(float), 32 * b0 + c2 + c3 + 1);
            }
#ifdef LOOP_VERSIONING
          } else {
            for (int c3 = 0; c3 <= 62; c3 += 1) {
              shared_table_1[c3][t0] = surf2Dread<float>(table, (x + c0) * sizeof(float), 32 * b0 + c2 + c3 + 1);
            }
          }
#endif
          }
          for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 2499); c3 += 1) {
#ifdef LOOP_VERSIONING_INNER
          if (-32 * b0 - c2 + 2499 <= 62) {
            for (int c4 = t0; c4 <= -32 * b0 - c2 + 2499; c4 += 32)
#else
            for (int c4 = t0; c4 <= ppcg_min(62, -32 * b0 - c2 + 2499); c4 += 32)
#endif
            {
              /* The only variant is c4, which only appears in the x index of table. */
              /* NOTE: Since we're using surface memory, all uses are affected. */
              // shared_table_5[c3][c4] = table[32 * b0 + c3][32 * b0 + c2 + c4];
              shared_table_5[c3][c4] = surf2Dread<float>(table, (32 * b0 + c2 + c4) * sizeof(float), 32 * b0 + c3);
            }
#ifdef LOOP_VERSIONING_INNER
          } else {
            for (int c4 = t0; c4 <= 62; c4 += 32) {
              shared_table_5[c3][c4] = surf2Dread<float>(table, (32 * b0 + c2 + c4) * sizeof(float), 32 * b0 + c3);
            }
          }
#endif
          }
          __syncthreads();
          if (x + c0 <= 2499 && c2 == 0) {
            /**
             * NOTE: The following part is rarely executed, providing little opportunity for optimization.
             */
            /* FIXME: Possibly accessing different rows of table, causing uncoalesced memory access. */
            // private_table_0[0][0] = ((private_table_0[0][0] >= table[x][x + c0 - 1]) ? private_table_0[0][0] : table[x][x + c0 - 1]);
            // private_table_0[0][0] = ((private_table_0[0][0] >= table[x + 1][x + c0]) ? private_table_0[0][0] : table[x + 1][x + c0]);
            // private_table_0[0][0] = ((private_table_0[0][0] >= (table[x + 1][x + c0 - 1] + (((seq[x] + seq[x + c0]) == 3) ? 1 : 0))) ? private_table_0[0][0] : (table[x + 1][x + c0 - 1] + (((seq[x] + seq[x + c0]) == 3) ? 1 : 0)));
            private_table_0[0][0] = ((private_table_0[0][0] >= surf2Dread<float>(table, (x + c0 - 1) * sizeof(float), x)) ? private_table_0[0][0] : surf2Dread<float>(table, (x + c0 - 1) * sizeof(float), x));
            private_table_0[0][0] = ((private_table_0[0][0] >= surf2Dread<float>(table, (x + c0) * sizeof(float), x + 1)) ? private_table_0[0][0] : surf2Dread<float>(table, (x + c0) * sizeof(float), x + 1));
            private_table_0[0][0] = ((private_table_0[0][0] >= (surf2Dread<float>(table, (x + c0 - 1) * sizeof(float), x + 1) + (((seq[x] + seq[x + c0]) == 3) ? 1 : 0))) ? private_table_0[0][0] : (surf2Dread<float>(table, (x + c0 - 1) * sizeof(float), x + 1) + (((seq[x] + seq[x + c0]) == 3) ? 1 : 0)));
          }
          if (x + c0 <= 2499) {
#ifdef LOOP_VERSIONING
          if (c0 - c2 - 1 <= 31) {
            for (int c4 = 0; c4 <= c0 - c2 - 1; c4 += 1)
#else
            for (int c4 = ppcg_max(0, -c2 + 1); c4 <= ppcg_min(31, c0 - c2 - 1); c4 += 1)
#endif
            {
              /* Accessing consecutive rows of the shared memory table; no bank conflicts. */
              private_table_0[0][0] = ((private_table_0[0][0] >= (shared_table_5[t0][t0 + c4] + shared_table_1[t0 + c4][t0])) ? private_table_0[0][0] : (shared_table_5[t0][t0 + c4] + shared_table_1[t0 + c4][t0]));
            }
#ifdef LOOP_VERSIONING
          } else {
            for (int c4 = 0; c4 <= 31; c4 += 1) {
              private_table_0[0][0] = ((private_table_0[0][0] >= (shared_table_5[t0][t0 + c4] + shared_table_1[t0 + c4][t0])) ? private_table_0[0][0] : (shared_table_5[t0][t0 + c4] + shared_table_1[t0 + c4][t0]));
            }
          }
#endif
          }
          __syncthreads();
        }
      } else {
        /**
         * NOTE: The following part is rarely executed, providing little opportunity for optimization.
         */
        if (x <= 2498) {
#ifdef LOOP_VERSIONING
        if (-32 * b0 + 2498 <= 62) {
          for (int c3 = 0; c3 <= -32 * b0 + 2498; c3 += 1)
#else
          for (int c3 = 0; c3 <= ppcg_min(62, -32 * b0 + 2498); c3 += 1)
#endif
          {
            /* The only variant is t0 (x), which only appears in the x index of table. */
            /* NOTE: Since we're using surface memory, all uses are affected. */
            // shared_table_1[c3][t0] = table[32 * b0 + c3 + 1][x + 1];
            shared_table_1[c3][t0] = surf2Dread<float>(table, (x + 1) * sizeof(float), 32 * b0 + c3 + 1);
          }
#ifdef LOOP_VERSIONING
        } else {
          for (int c3 = 0; c3 <= 62; c3 += 1) {
            shared_table_1[c3][t0] = surf2Dread<float>(table, (x + 1) * sizeof(float), 32 * b0 + c3 + 1);
          }
        }
#endif
        }
        for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 2499); c3 += 1) {
#ifdef LOOP_VERSIONING_INNER
        if (-32 * b0 + 2499 <= 62) {
          for (int c4 = t0; c4 <= -32 * b0 + 2499; c4 += 32)
#else
          for (int c4 = t0; c4 <= ppcg_min(62, -32 * b0 + 2499); c4 += 32)
#endif
          {
            /* The only variant is c4, which only appears in the x index of table. */
            /* NOTE: Since we're using surface memory, all uses are affected. */
            // shared_table_5[c3][c4] = table[32 * b0 + c3][32 * b0 + c4];
            shared_table_5[c3][c4] = surf2Dread<float>(table, (32 * b0 + c4) * sizeof(float), 32 * b0 + c3);
          }
#ifdef LOOP_VERSIONING_INNER
        } else {
          for (int c4 = t0; c4 <= 62; c4 += 32) {
            shared_table_5[c3][c4] = surf2Dread<float>(table, (32 * b0 + c4) * sizeof(float), 32 * b0 + c3);
          }
        }
#endif
        }
        __syncthreads();
        if (x <= 2498) {
          /* FIXME: Possibly accessing different rows of table, causing uncoalesced memory access. */
          // private_table_0[0][0] = ((private_table_0[0][0] >= table[x][x]) ? private_table_0[0][0] : table[x][x]);
          // private_table_0[0][0] = ((private_table_0[0][0] >= table[x + 1][x + 1]) ? private_table_0[0][0] : table[x + 1][x + 1]);
          private_table_0[0][0] = ((private_table_0[0][0] >= surf2Dread<float>(table, x * sizeof(float), x)) ? private_table_0[0][0] : surf2Dread<float>(table, x * sizeof(float), x));
          private_table_0[0][0] = ((private_table_0[0][0] >= surf2Dread<float>(table, (x + 1) * sizeof(float), x + 1)) ? private_table_0[0][0] : surf2Dread<float>(table, (x + 1) * sizeof(float), x + 1));
          private_table_0[0][0] = ((private_table_0[0][0] >= private_table_6[0][0]) ? private_table_0[0][0] : private_table_6[0][0]);
        }
        __syncthreads();
      }
      /* NOTE: A memory address is only written once at the end of the kernel; no read-after-write. */
      if (x + c0 <= 2499) {
        /* FIXME: Accessing different rows of table, causing uncoalesced memory access. */
        // table[x][x + c0] = private_table_0[0][0];
        surf2Dwrite(private_table_0[0][0], table, (x + c0) * sizeof(float), x);
      }
    }
}
