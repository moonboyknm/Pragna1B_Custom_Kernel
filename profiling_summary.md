# Profiling Summary: Triton RMSNorm Kernel

**Target Device:** NVIDIA GeForce RTX 2050 (4GB VRAM)
**Date:** February 13, 2026

## 1. Executive Performance Summary

The kernel is heavily **memory-bound**, which is characteristic of the RMSNorm operation in Small Language Models (SLMs) like Pragna-1B. It successfully saturates the RTX 2050's memory bandwidth, leaving little room for optimization without architectural changes like manual memory coalescing in C++.

| Metric | Value | Baseline Analysis |
| --- | --- | --- |
| **DRAM Throughput** | **92.34%** | Excellent utilization of available VRAM bandwidth. |
| **SM Throughput** | **8.92%** | Low compute intensity; SMs are mostly waiting for data. |
| **Execution Time** | **100.22 µs** | The target time to beat with custom C++ CUDA kernels. |
| **Achieved Occupancy** | **93.52%** | High warp parallelism effectively hiding latency. |

## 2. Hardware Bottleneck Analysis

Nsight Compute identified three primary areas where hardware performance is limited:

* **Memory Bound (SOL)**: The kernel is limited by the **DRAM workload**, utilizing >80% of peak performance. The compute pipelines are significantly under-utilized.
* **Warp Stall Reasons**: Warps are stalled **45.9% of the time** due to **L1TEX scoreboard dependencies**. This indicates that the execution units are frequently waiting for memory loads from the L1 cache to complete.
* **Instruction Pipeline**: The kernel executes a mix of fused and non-fused FP32 instructions. Converting pairs of non-fused instructions to fused, higher-throughput equivalents (like FMA) could potentially increase FP32 performance by up to **33%**.

## 3. Optimization Targets for C++ CUDA (PBL Goals)

To outperform this Triton implementation in your Project-Based Learning (PBL) work, focus on the following strategies:

1. **Manual Coalescing**: Use `float4` vector loads to ensure the hardware can group memory requests even more efficiently than the Triton compiler.
2. **Shared Memory Tiling**: Experiment with loading row data into **Shared Memory (SRAM)** to reduce the L1TEX scoreboard stalls identified in the profiler.
3. **Strict Instruction Fusion**: Manually implement fused multiply-add operations to target the 33% throughput gap noted in the Instruction Statistics.
