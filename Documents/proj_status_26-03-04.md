# Brief
Alright so I've profiled the triton kernel for pragna-1b and now my goal is to learn how to make a better kernel in cuda.

Now I've got to figure out the goals for the rest of this learning/building project.

(Taking in mind the popular advice from the GPU Mode community, I should try to learn everything about this project related to CUDA and C++)

# Goals

**1. Implement Manual Memory Coalescing**

* **The Target:** The RMSNorm kernel is currently heavily memory-bound and saturates the available VRAM bandwidth. To beat Triton, you need to use `float4` vector loads to group memory requests more efficiently than the compiler does.

* **What to Learn:** Master the alignment constraints of CUDA memory coalescing. In C++, practice safely casting standard float pointers to `float4` types to pull 128 bits per thread in a single hardware instruction, minimizing DRAM transactions.

**2. Apply Shared Memory Tiling**

* **The Target:** The profiler identified that warps are stalling 45.9% of the time due to L1TEX scoreboard dependencies, meaning execution units are sitting idle waiting for L1 cache loads. Your goal is to load row data into Shared Memory (SRAM) to eliminate these specific stalls.

* **What to Learn:** Dive deeply into the GPU memory hierarchy. Focus on how to declare `__shared__` memory arrays, utilize block-level synchronization (`__syncthreads()`) to ensure data is ready before computation, and structure your thread access patterns to avoid shared memory bank conflicts.

**3. Enforce Strict Instruction Fusion**

* **The Target:** The current implementation uses a mix of fused and non-fused FP32 instructions. By manually forcing fused multiply-add operations, you can target a potential 33% increase in FP32 throughput.

* **What to Learn:** Get comfortable using hardware math intrinsics like `fmaf()` in C++. To validate these low-level optimizations for AI infrastructure hardware, learn how to inspect the generated PTX and SASS assembly code to verify that `nvcc` actually emitted the fused instructions instead of breaking them apart.

The overarching theme of this project will be shifting the hardware bottlenecks. Because RMSNorm on Small Language Models (SLMs) like Pragna-1B drastically under-utilizes compute pipelines while waiting on DRAM, mastering these specific C++ memory control and instruction-level techniques will be the deciding factor in your kernel's performance.
