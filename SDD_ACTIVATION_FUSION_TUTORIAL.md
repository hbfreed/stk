# SDD Activation Fusion Tutorial

This tutorial walks through fusing activation functions into the SDD (Sparse = Dense × Dense) kernel in STK.

## 1. Background: Why SDD, Not DSD?

In the megablocks MLP pattern, the data flow is:

```python
x_permuted = stk.ops.sdd(x_permuted, self.w1, topology)  # dense × dense -> SPARSE
x_permuted = gelu(x_permuted)                            # activation on SPARSE
x_permuted = stk.ops.dsd(x_permuted, self.w2)            # sparse × dense -> dense
```

The activation is applied to the **sparse output of SDD**, not the dense output of DSD. This matters because:

1. **Smaller tensor**: The sparse `.data` tensor only contains non-zero blocks, much smaller than a full dense matrix
2. **Zeros preserved**: All common activations pass through zero:
   - `relu(0) = 0`
   - `relu²(0) = 0`
   - `silu(0) = 0 × sigmoid(0) = 0`
   - `gelu(0) = 0`

So applying activation only to sparse blocks is mathematically equivalent to applying it to the full dense matrix.

## 2. The SDD Kernel

### Location
`stk/backend/triton_kernels.py`, lines 33-68

### How It Works

```python
@triton.jit
def _sdd_kernel(A, B, C, M, N, K, ...):
    # Each program handles ONE sparse block
    pid = tl.program_id(0)
    pid_m = tl.load(row_indices + pid)     # Which block-row
    pid_n = tl.load(column_indices + pid)  # Which block-col

    # Compute block matmul
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)  # fp32
    for k in range(...):
        a = tl.load(A_ptr)
        b = tl.load(B_ptr)
        acc += tl.dot(a, b)

    # Convert and store
    acc = acc.to(C.dtype.element_ty)  # fp32 -> fp16
    tl.store(C_ptr, acc)
```

### Fusion Point

Fuse activation **before `tl.store()`**, while the accumulator is still in registers:

```python
    # Accumulator is fp32 here - good precision for activation

    # === FUSE ACTIVATION HERE ===
    if ACTIVATION == "relu":
        acc = tl.maximum(acc, 0.0)
    elif ACTIVATION == "relu_squared":
        acc = tl.maximum(acc, 0.0)
        acc = acc * acc
    elif ACTIVATION == "silu":
        acc = acc * tl.sigmoid(acc)
    elif ACTIVATION == "gelu":
        tanh_arg = 0.7978845608 * (acc + 0.044715 * acc * acc * acc)
        tanh_val = 2.0 * tl.sigmoid(2.0 * tanh_arg) - 1.0
        acc = 0.5 * acc * (1.0 + tanh_val)
    # =============================

    acc = acc.to(C.dtype.element_ty)  # Convert AFTER activation
    tl.store(C_ptr, acc)
```

**Key points:**
- Apply activation while `acc` is still fp32 for numerical stability
- Convert to output dtype (fp16) after activation
- No extra memory traffic - activation is "free" in terms of bandwidth

## 3. Forward Pass Changes

### 3.1 Kernel Signature

Add `ACTIVATION: tl.constexpr` parameter:

```python
@triton.autotune(
    configs=[...],
    key=['M', 'N', 'K'],  # Don't need ACTIVATION in key
)
@triton.jit
def _sdd_kernel(A, B, C, M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                row_indices, column_indices,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                BLOCK_SIZE: tl.constexpr, GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
                ACTIVATION: tl.constexpr = "none",  # <-- ADD THIS
                ):
```

### 3.2 Launcher Function

Update `sdd()` launcher in `triton_kernels.py`:

```python
def sdd(lhs, rhs, shape, out, offsets, row_indices, column_indices, activation="none"):
    # ... setup code ...

    _sdd_kernel[grid](
        lhs, rhs, out, M, N, K,
        # ... strides ...
        row_indices, column_indices,
        GROUP_M=128, ACC_TYPE=ACC_TYPE,
        ACTIVATION=activation  # <-- ADD THIS
    )
```

### 3.3 Autograd Wrapper

Update `SDD` class in `sputnik.py`:

```python
class SDD(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, lhs, rhs, shape, data, offsets, row_indices,
                column_indices, offsets_t, column_indices_t, block_offsets_t,
                activation="none"):  # <-- ADD PARAMETER

        ctx.activation = activation
        # ... existing save_for_backward ...

        out = torch.empty(data.shape, dtype=lhs.dtype, device=lhs.device)

        backend.sdd(lhs, rhs, shape, out, offsets, row_indices,
                    column_indices, activation=activation)  # <-- PASS IT

        return out
```

### 3.4 User API

Update `sdd()` in `linear_ops.py`:

```python
def sdd(a, b, topo, activation="none"):
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert isinstance(topo, Matrix)

    out = sputnik.sdd(
        a, b,
        topo.size(),
        topo.data, topo.offsets,
        topo.row_indices, topo.column_indices,
        topo.offsets_t, topo.column_indices_t, topo.block_offsets_t,
        activation)  # <-- ADD THIS

    return Matrix(topo.size(), out, ...)
```

## 4. Backward Pass Changes

The SDD backward uses **DDS and DSD kernels**, not the SDD kernel:
- `dlhs` (gradient w.r.t. left input): computed via DDS
- `drhs` (gradient w.r.t. right input): computed via DSD

So the activation gradient is computed **separately** from the matmul gradient.

### Option A: Save Pre-Activation (simple but uses memory)

```python
def forward(ctx, ..., activation="none"):
    # Compute matmul without activation
    backend.sdd(..., activation="none")

    if activation != "none":
        pre_act = out.clone()  # or just reference if we allocate new out
        if activation == "relu":
            out = torch.relu(out)
        # ... etc
        ctx.save_for_backward(..., pre_act)
    else:
        ctx.save_for_backward(...)

def backward(ctx, dy):
    if ctx.activation != "none":
        # Multiply dy by activation derivative
        dy = compute_activation_grad(dy, pre_act, ctx.activation)
    # Then compute matmul gradients with modified dy
```

### Option B: Use PyTorch aten ops (faster)

```python
def backward(ctx, dy):
    if ctx.activation == "relu":
        dy = torch.ops.aten.threshold_backward(dy, saved_act, 0.0)
    elif ctx.activation == "silu":
        dy = torch.ops.aten.silu_backward(dy, saved_act)
    elif ctx.activation == "gelu":
        dy = torch.ops.aten.gelu_backward(dy, saved_act, approximate='tanh')
    # ... then matmul gradients
```

## 5. Testing

### Correctness Test

```python
import torch
import stk

# Create sparse matrix and dense inputs
sparse_topo = stk.random.mask(M, N, sparsity=0.9, blocking=128).cuda()
a = torch.randn(M, K, device='cuda', dtype=torch.float16, requires_grad=True)
b = torch.randn(K, N, device='cuda', dtype=torch.float16, requires_grad=True)

# Unfused reference
out_unfused = stk.ops.sdd(a, b, sparse_topo)
out_unfused.data = torch.relu(out_unfused.data) ** 2  # apply activation to sparse data
loss_unfused = out_unfused.data.sum()
loss_unfused.backward()
grad_a_unfused = a.grad.clone()

# Fused
a.grad = None
b.grad = None
out_fused = stk.ops.sdd(a, b, sparse_topo, activation="relu_squared")
loss_fused = out_fused.data.sum()
loss_fused.backward()
grad_a_fused = a.grad.clone()

# Compare
assert torch.allclose(out_fused.data, out_unfused.data, rtol=0.05, atol=0.1)
assert torch.allclose(grad_a_fused, grad_a_unfused, rtol=0.1, atol=0.1)
```

### Benchmark

```python
import time

def bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return time.time() - start

# Unfused
t_unfused = bench(lambda: torch.relu(stk.ops.sdd(a, b, topo).data) ** 2)

# Fused
t_fused = bench(lambda: stk.ops.sdd(a, b, topo, activation="relu_squared"))

print(f"Speedup: {t_unfused / t_fused:.2f}x")
```

## 6. Summary

| Step | File | Change |
|------|------|--------|
| 1 | `triton_kernels.py` | Add `ACTIVATION` to `_sdd_kernel`, apply before store |
| 2 | `triton_kernels.py` | Add `activation` param to `sdd()` launcher |
| 3 | `sputnik.py` | Add `activation` to `SDD.forward()`, save for backward |
| 4 | `sputnik.py` | Handle activation gradient in `SDD.backward()` |
| 5 | `linear_ops.py` | Add `activation` param to user-facing `sdd()` |

The key difference from the previous (incorrect) DSD approach:
- SDD outputs **sparse** data, which is smaller
- Activation on sparse data = activation only on non-zero blocks
- This is valid because all activations preserve zeros
