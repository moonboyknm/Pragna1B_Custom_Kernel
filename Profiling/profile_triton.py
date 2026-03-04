import torch
import triton
import triton.language as tl

# 1. THE KERNEL DEFINITION
@triton.jit
def rms_norm_kernel(
    X,  # pointer to input
    Y,  # pointer to output
    W,  # pointer to weights (gamma)
    stride,  # how much to skip to get to the next row
    N_COLS,  # width of the hidden dimension
    eps,  # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    # Map program to a specific row
    row_idx = tl.program_id(0)
    row_start_ptr = X + row_idx * stride
    
    # Load the row into SRAM
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_COLS
    x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Calculate RMS: sqrt(mean(x^2) + eps)
    # Triton's 'tl.sum' is highly optimized for the RTX 2050's SMs
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / N_COLS
    rrms = 1 / tl.sqrt(var + eps)
    
    # Apply normalization and weights
    w = tl.load(W + offsets, mask=mask, other=0.0).to(tl.float32)
    y = (x * rrms).to(tl.float16) * w
    
    # Write back to VRAM
    y_ptr = Y + row_idx * stride
    tl.store(y_ptr + offsets, y, mask=mask)

# 2. THE WRAPPER FUNCTION
def rms_norm_triton(x, weight, eps=1e-6):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Grid is 1D: one program per row
    grid = (M,)
    
    rms_norm_kernel[grid](
        x, y, weight,
        x.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

# 3. BENCHMARKING & WARM-UP
if __name__ == "__main__":
    # Match Pragna-1B / Llama hidden size (e.g., 2048 or 4096)
    M, N = 1024, 2048 
    x = torch.randn((M, N), device='cuda', dtype=torch.float16)
    w = torch.ones(N, device='cuda', dtype=torch.float16)

    # WARM-UP: Crucial for Nsight Compute to ignore JIT compilation time
    print("Warming up kernel...")
    for _ in range(20):
        _ = rms_norm_triton(x, w)

    # PROFILED EXECUTION
    # This is the call ncu will actually record
    print(f"Profiling Triton RMSNorm (Grid: {M}, Block: {triton.next_power_of_2(N)})")
    result = rms_norm_triton(x, w)
    
    torch.cuda.synchronize()
    print("Done.")
