#include "floyd-warshall_kernel.hu"
__global__ void kernel0(cudaSurfaceObject_t path, int c0, int c1)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int x = 32 * b0 + t0;

    if (x <= 2799 && c1 >= x && x + 2799 >= c1) {
//    path[x][c1 - x] = (
//      (path[x][c1 - x] < (path[x][c0] + path[c0][c1 - x]))
//        ? path[x][c1 - x]
//        : (path[x][c0] + path[c0][c1 - x])
//    );
      /* NOTE: A location is only write once at the end of the kernel, so we can use surf2Dwrite. */
      surf2Dwrite<float>(
        surf2Dread<float>(path, (c1 - x) * sizeof(float), x) < (surf2Dread<float>(path, c0 * sizeof(float), x) + surf2Dread<float>(path, (c1 - x) * sizeof(float), c0))
          ? surf2Dread<float>(path, (c1 - x) * sizeof(float), x)
          : (surf2Dread<float>(path, c0 * sizeof(float), x) + surf2Dread<float>(path, (c1 - x) * sizeof(float), c0)),
        path,
        (c1 - x) * sizeof(float),
        x
      );
    }
}
