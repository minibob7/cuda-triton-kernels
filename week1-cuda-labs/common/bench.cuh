// ============================================================
// bench.cuh —— 所有实验共用的计时/校验工具
// 只有一个头文件，#include 即可，无需额外编译
// ============================================================
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// 所有 CUDA API 调用都套这个宏：出错立刻打印文件行号并退出。
// 养成习惯：CUDA 错误不会抛异常，不检查的话错误会被“吞”到后面某一行才爆。
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// 用 CUDA Event 计时：这是 GPU 计时的标准做法
// （CPU 的 clock() 量不到异步执行的 kernel）
struct GpuTimer {
    cudaEvent_t beg, end;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&beg)); CUDA_CHECK(cudaEventCreate(&end)); }
    ~GpuTimer() { CUDA_CHECK(cudaEventDestroy(beg)); CUDA_CHECK(cudaEventDestroy(end)); }
    void start() { CUDA_CHECK(cudaEventRecord(beg)); }
    void stop()  { CUDA_CHECK(cudaEventRecord(end)); CUDA_CHECK(cudaEventSynchronize(end)); }
    float ms() const {
        float m = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&m, beg, end));
        return m;
    }
};

// 标准基准流程：先 warmup（触发上下文/缓存/自动调优），
// 再重复 reps 次，取中位数（比平均值抗抖动）。
// fn 是一个 lambda，里面放一次完整的 kernel launch。
template <class F>
float bench_ms(F&& fn, int warmup = 3, int reps = 20) {
    GpuTimer t;
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> ts;
    ts.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        t.start(); fn(); t.stop();
        ts.push_back(t.ms());
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

// 对账工具：device 结果 vs CPU 参考，打印最大误差
// （浮点顺序不同会有微小误差，看量级而不是苛求为 0）
inline void verify(const char* name, const float* got, const float* ref, int n) {
    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = fabs((double)got[i] - (double)ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("[%s] max_err = %.6g  %s\n", name, max_err, max_err < 1e-3 ? "OK" : "FAIL");
}
