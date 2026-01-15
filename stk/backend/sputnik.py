import torch

from stk.backend import triton_kernels as backend
from stk.backend.autocast import custom_bwd, custom_fwd


def _standardize_shape(x, transpose):
    if transpose:
        return torch.Size((x[1], x[0]))
    return x


def _sparse_transpose(x):
    return (torch.Size((x[0][1], x[0][0])), ) + x[1:]


def _transpose_helper(x, transpose):
    if isinstance(x, torch.Tensor):
        return x.t() if transpose else x
    if transpose:
        x = _sparse_transpose(x)
    return x + (transpose,)


def _wrap(x):
    if isinstance(x, torch.Tensor):
        return (x,)
    return x


def _is_transposed(x):
    return (not x.is_contiguous() and
            x.stride()[0] == 1 and
            x.stride()[1] == x.size()[0])


def _call_helper(op, out, a, b, trans_a, trans_b):
    args = (_wrap(_transpose_helper(a, trans_a)) +
            _wrap(_transpose_helper(b, trans_b)))
    if isinstance(out, tuple):
        args = args + out
    return op(*args)


def _preprocess_inputs(lhs, rhs, dy):
    if isinstance(lhs, torch.Tensor) and _is_transposed(lhs):
        lhs = lhs.t()
    if isinstance(rhs, torch.Tensor) and _is_transposed(rhs):
        rhs = rhs.t()
    if (isinstance(dy, torch.Tensor) and
        not dy.is_contiguous() and
        not _is_transposed(dy)):
        dy = dy.contiguous()
    if isinstance(dy, tuple) and not dy[1].is_contiguous():
        dy = (dy[0], dy[1].contiguous()) + dy[2:]
    return lhs, rhs, dy


def _postprocess_outputs(x, transpose, grad):
    if isinstance(x, torch.Tensor) and transpose:
        return grad.t()
    return grad


def _lhs_gradient(op, lhs, rhs, dy, trans_lhs, trans_rhs):
    lhs, rhs, dy = _preprocess_inputs(lhs, rhs, dy)

    a, b = (rhs, dy) if trans_lhs else (dy, rhs)
    trans_a = trans_lhs and trans_rhs
    trans_b = trans_lhs or not trans_rhs
    out = _call_helper(op, lhs, a, b, trans_a, trans_b)
    return _postprocess_outputs(lhs, trans_lhs, out)


def _rhs_gradient(op, lhs, rhs, dy, trans_lhs, trans_rhs):
    lhs, rhs, dy = _preprocess_inputs(lhs, rhs, dy)

    a, b = (dy, lhs) if trans_rhs else (lhs, dy)
    trans_a = not trans_lhs or trans_rhs
    trans_b = trans_lhs and trans_rhs
    out = _call_helper(op, rhs, a, b, trans_a, trans_b)
    return _postprocess_outputs(rhs, trans_rhs, out)


class DSD(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx,
                shape,
                data,
                offsets,
                row_indices,
                column_indices,
                offsets_t,
                column_indices_t,
                block_offsets_t,
                transpose_a,
                rhs,
                activation="none"):
        ctx.shape = _standardize_shape(shape, transpose_a)
        ctx.transpose_a = transpose_a
        ctx.activation = activation

        out = torch.empty(
            (shape[0], rhs.size()[1]),
            dtype=rhs.dtype,
            device=rhs.device)

        if activation == "none":
            # No activation - just matmul
            backend.dsd(shape, data, offsets, row_indices, column_indices,
                        offsets_t, column_indices_t, block_offsets_t,
                        transpose_a, rhs, out, activation="none")
            ctx.save_for_backward(data, offsets, row_indices, column_indices,
                                  offsets_t, column_indices_t, block_offsets_t, rhs)
        elif activation == "relu":
            # ReLU: use fused kernel, save output (output > 0 tells us gradient mask)
            backend.dsd(shape, data, offsets, row_indices, column_indices,
                        offsets_t, column_indices_t, block_offsets_t,
                        transpose_a, rhs, out, activation="relu")
            ctx.save_for_backward(data, offsets, row_indices, column_indices,
                                  offsets_t, column_indices_t, block_offsets_t, rhs, out)
        elif activation == "relu_squared":
            # ReLU²: compute matmul, save pre_act, apply activation
            backend.dsd(shape, data, offsets, row_indices, column_indices,
                        offsets_t, column_indices_t, block_offsets_t,
                        transpose_a, rhs, out, activation="none")
            pre_act = out  # keep reference
            out = torch.relu(pre_act) ** 2
            ctx.save_for_backward(data, offsets, row_indices, column_indices,
                                  offsets_t, column_indices_t, block_offsets_t, rhs, pre_act)
        else:
            # silu/gelu: compute matmul, save pre_act, apply activation in PyTorch
            backend.dsd(shape, data, offsets, row_indices, column_indices,
                        offsets_t, column_indices_t, block_offsets_t,
                        transpose_a, rhs, out, activation="none")
            pre_act = out  # keep reference
            if activation == "silu":
                out = torch.nn.functional.silu(pre_act)
            elif activation == "gelu":
                out = torch.nn.functional.gelu(pre_act, approximate='tanh')
            ctx.save_for_backward(data, offsets, row_indices, column_indices,
                                  offsets_t, column_indices_t, block_offsets_t, rhs, pre_act)

        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        saved_tensors = ctx.saved_tensors
        trans_a = ctx.transpose_a

        if ctx.activation == "none":
            lhs = (ctx.shape,) + saved_tensors[:-1]
            rhs = saved_tensors[-1]
        else:
            # Last tensor is pre_act (or out for relu), second to last is rhs
            lhs = (ctx.shape,) + saved_tensors[:-2]
            rhs = saved_tensors[-2]
            saved_act = saved_tensors[-1]

            if ctx.activation == "relu":
                # saved_act is the output; use it to determine where pre_act > 0
                dy = dy * (saved_act > 0).to(dy.dtype)
            elif ctx.activation == "relu_squared":
                # saved_act is pre_act
                dy = dy * 2 * torch.clamp(saved_act, min=0)
            elif ctx.activation == "silu":
                sig = torch.sigmoid(saved_act)
                dy = dy * (sig + saved_act * sig * (1 - sig))
            elif ctx.activation == "gelu":
                k = 0.7978845608
                x3 = saved_act ** 3
                tanh_arg = k * (saved_act + 0.044715 * x3)
                tanh_val = torch.tanh(tanh_arg)
                sech2 = 1 - tanh_val ** 2
                dy = dy * (0.5 * (1 + tanh_val) + 0.5 * saved_act * sech2 * k * (1 + 0.134145 * saved_act ** 2))

        trans_b = _is_transposed(rhs)

        ddata = None
        if ctx.needs_input_grad[1]:
            ddata = _lhs_gradient(sdd,
                                  lhs,
                                  rhs,
                                  dy,
                                  trans_a,
                                  trans_b)
        drhs = None
        if ctx.needs_input_grad[9]:  # rhs
            op = dds if trans_b else dsd
            drhs = _rhs_gradient(op,
                                 lhs,
                                 rhs,
                                 dy,
                                 trans_a,
                                 trans_b)
        return None, ddata, None, None, None, None, None, None, None, drhs, None


dsd = DSD.apply


class DDS(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx,
                lhs,
                shape,
                data,
                offsets,
                row_indices,
                column_indices,
                offsets_t,
                column_indices_t,
                block_offsets_t,
                transpose_b):
        ctx.save_for_backward(lhs,
                              data,
                              offsets,
                              row_indices,
                              column_indices,
                              offsets_t,
                              column_indices_t,
                              block_offsets_t)
        ctx.shape = _standardize_shape(shape, transpose_b)
        ctx.transpose_b = transpose_b
        out = torch.empty((lhs.size()[0], shape[1]),
                          dtype=lhs.dtype,
                          device=lhs.device)
        backend.dds(lhs,
                    shape,
                    data,
                    offsets,
                    row_indices,
                    column_indices,
                    offsets_t,
                    column_indices_t,
                    block_offsets_t,
                    transpose_b,
                    out)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        saved_tensors = ctx.saved_tensors
        lhs = saved_tensors[0]
        rhs = (ctx.shape,) + saved_tensors[1:]
        trans_a = _is_transposed(lhs)
        trans_b = ctx.transpose_b

        dlhs = None
        if ctx.needs_input_grad[0]:
            op = dsd if trans_a else dds
            dlhs = _lhs_gradient(op,
                                 lhs,
                                 rhs,
                                 dy,
                                 trans_a,
                                 trans_b)
        ddata = None
        if ctx.needs_input_grad[2]:
            ddata = _rhs_gradient(sdd,
                                  lhs,
                                  rhs,
                                  dy,
                                  trans_a,
                                  trans_b)
        return dlhs, None, ddata, None, None, None, None, None, None, None


dds = DDS.apply


class SDD(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx,
                lhs,
                rhs,
                shape,
                data,
                offsets,
                row_indices,
                column_indices,
                offsets_t,
                column_indices_t,
                block_offsets_t):
        ctx.save_for_backward(
            lhs,
            rhs,
            offsets,
            row_indices,
            column_indices,
            offsets_t,
            column_indices_t,
            block_offsets_t)
        ctx.shape = shape
        out = torch.empty(
            data.shape,
            dtype=lhs.dtype,
            device=lhs.device)
        backend.sdd(lhs,
                    rhs,
                    shape,
                    out,
                    offsets,
                    row_indices,
                    column_indices)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        saved_tensors = ctx.saved_tensors
        lhs, rhs = saved_tensors[:2]
        dy = (ctx.shape, dy) + saved_tensors[2:]
        trans_a = _is_transposed(lhs)
        trans_b = _is_transposed(rhs)

        dlhs = None
        if ctx.needs_input_grad[0]:
            op = dds if trans_a else dsd
            dlhs = _lhs_gradient(op,
                                 lhs,
                                 rhs,
                                 dy,
                                 trans_a,
                                 trans_b)
        drhs = None
        if ctx.needs_input_grad[1]:
            op = dsd if trans_b else dds
            drhs = _rhs_gradient(op,
                                 lhs,
                                 rhs,
                                 dy,
                                 trans_a,
                                 trans_b)
        return dlhs, drhs, None, None, None, None, None, None, None, None


sdd = SDD.apply

class RowIndices(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shape, data, offsets, column_indices):
        out = torch.empty(
            column_indices.shape,
            dtype=column_indices.dtype,
            device=column_indices.device)
        backend.row_indices(shape, data, offsets, column_indices, out)
        return out


row_indices = RowIndices.apply
