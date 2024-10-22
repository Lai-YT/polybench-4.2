#ifndef TIMER_HPP
#define TIMER_HPP

#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <time.h>

enum class TimeUnit { S, Ms, Us, Ns };

#define cudaCheckReturn(ret)                               \
  do {                                                     \
    cudaError_t cudaCheckReturn_e = (ret);                 \
    if (cudaCheckReturn_e != cudaSuccess) {                \
      std::fprintf(stderr, "CUDA error: %s\n",             \
                   cudaGetErrorString(cudaCheckReturn_e)); \
      std::fflush(stderr);                                 \
    }                                                      \
    assert(cudaCheckReturn_e == cudaSuccess);              \
  } while (0)
#define cudaCheckKernel()                \
  do {                                   \
    cudaCheckReturn(cudaGetLastError()); \
  } while (0)

class GpuEventTimer {
public:
  GpuEventTimer() {
    cudaCheckReturn(cudaEventCreate(&start));
    cudaCheckReturn(cudaEventCreate(&stop));
  }
  void Start() { cudaCheckReturn(cudaEventRecord(start, 0)); }
  /**
   * @note The start time is automatically reused if two stops are called
   * without a start in between.
   */
  void Stop() { cudaCheckReturn(cudaEventRecord(stop, 0)); }
  /**
   * @return Elapsed time in the specified time unit.
   * @note A `Stop` must be called before calling `ElapsedTime`.
   */
  template <TimeUnit unit = TimeUnit::Ms> double ElapsedTime() {
    double elapsed = ElapsedTimeMs_();
    switch (unit) {
    case TimeUnit::S:
      return elapsed * 1e-3;
    case TimeUnit::Ms:
      return elapsed;
    case TimeUnit::Us:
      return elapsed * 1e3;
    case TimeUnit::Ns:
      return elapsed * 1e6;
    }
    assert(false && "Unknown time unit");
    return 0; // to suppress warning
  }

private:
  cudaEvent_t start, stop;

  float ElapsedTimeMs_() {
    cudaCheckReturn(cudaEventSynchronize(stop));
    float milliseconds = 0;
    cudaCheckReturn(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
  }
};

class CpuTimer {
public:
  void Start() { clock_gettime(CLOCK_MONOTONIC, &start); }
  void Stop() { clock_gettime(CLOCK_MONOTONIC, &stop); }

  /**
   * @return Elapsed time in the specified time unit.
   * @note A `Stop` must be called before calling `ElapsedTime`.
   */
  template <TimeUnit unit = TimeUnit::Ms> double ElapsedTime() {
    double elapsed = ElapsedTimeMs_();
    switch (unit) {
    case TimeUnit::S:
      return elapsed * 1e-3;
    case TimeUnit::Ms:
      return elapsed;
    case TimeUnit::Us:
      return elapsed * 1e3;
    case TimeUnit::Ns:
      return elapsed * 1e6;
    }
    assert(false && "Unknown time unit");
    return 0; // to suppress warning
  }

private:
  struct timespec start, stop;

  double ElapsedTimeMs_() {
    return (stop.tv_sec - start.tv_sec) * 1e3 +
           (stop.tv_nsec - start.tv_nsec) * 1e-6;
  }
};

#undef cudaCheckReturn
#undef cudaCheckKernel

#endif // TIMER_HPP
