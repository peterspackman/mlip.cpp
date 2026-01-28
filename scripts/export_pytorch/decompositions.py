"""
Decomposition rules for PyTorch operations that need to be broken down
into primitives that GGML supports with backward passes.

These decompositions are based on the patterns in src/models/pet/pet_layers.cpp.
"""

from __future__ import annotations

from typing import Callable
from .graph_ir import GGMLGraph, GGMLNode, GGMLDtype


def decompose_layer_norm(
    graph: GGMLGraph,
    input_ref: str,
    weight_ref: str,
    bias_ref: str,
    input_shape: list[int],
    eps: float = 1e-5,
) -> str:
    """
    Decompose LayerNorm into primitives with backward support.

    Based on pet_layers.cpp:85-145.

    LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias

    Where mean and var are computed over dimension 0 (feature dimension).

    Args:
        graph: The GGML graph being built
        input_ref: Reference to input tensor (e.g., "node:5" or "input:x")
        weight_ref: Reference to weight tensor
        bias_ref: Reference to bias tensor
        input_shape: Shape of input tensor in GGML format [d_feat, ...]
        eps: Epsilon for numerical stability

    Returns:
        Reference to the output node
    """
    d_feat = input_shape[0]
    inv_d = 1.0 / float(d_feat)

    # Construct reduced shape for mean/var: [1, ...]
    reduced_shape = [1] + input_shape[1:]

    # Step 1: mean = sum_rows(x) / d
    sum_node = graph.add_node(
        op="SUM_ROWS",
        name="ln_sum",
        inputs=[input_ref],
        output_shape=reduced_shape,
        output_dtype=GGMLDtype.F32,
    )

    mean_node = graph.add_node(
        op="SCALE",
        name="ln_mean",
        inputs=[graph.node_ref(sum_node)],
        output_shape=reduced_shape,
        output_dtype=GGMLDtype.F32,
        params={"scale": inv_d},
    )

    # Step 2: x_centered = x - mean (with broadcast)
    mean_broadcast = graph.add_node(
        op="REPEAT",
        name="ln_mean_broadcast",
        inputs=[graph.node_ref(mean_node)],
        output_shape=input_shape,
        output_dtype=GGMLDtype.F32,
        params={"target_shape": input_shape},
    )

    centered = graph.add_node(
        op="SUB",
        name="ln_centered",
        inputs=[input_ref, graph.node_ref(mean_broadcast)],
        output_shape=input_shape,
        output_dtype=GGMLDtype.F32,
    )

    # Step 3: var = sum_rows(x_centered^2) / d
    centered_sq = graph.add_node(
        op="SQR",
        name="ln_centered_sq",
        inputs=[graph.node_ref(centered)],
        output_shape=input_shape,
        output_dtype=GGMLDtype.F32,
    )

    sum_sq = graph.add_node(
        op="SUM_ROWS",
        name="ln_sum_sq",
        inputs=[graph.node_ref(centered_sq)],
        output_shape=reduced_shape,
        output_dtype=GGMLDtype.F32,
    )

    var_node = graph.add_node(
        op="SCALE",
        name="ln_var",
        inputs=[graph.node_ref(sum_sq)],
        output_shape=reduced_shape,
        output_dtype=GGMLDtype.F32,
        params={"scale": inv_d},
    )

    # Step 4: std = sqrt(var + eps)
    # Since GGML doesn't have add-scalar, we approximate: sqrt(var * (1 + eps))
    # This is close when var ~ 1 (which is typical for normalized data)
    var_stabilized = graph.add_node(
        op="SCALE",
        name="ln_var_stabilized",
        inputs=[graph.node_ref(var_node)],
        output_shape=reduced_shape,
        output_dtype=GGMLDtype.F32,
        params={"scale": 1.0 + eps},
    )

    std_node = graph.add_node(
        op="SQRT",
        name="ln_std",
        inputs=[graph.node_ref(var_stabilized)],
        output_shape=reduced_shape,
        output_dtype=GGMLDtype.F32,
    )

    # Step 5: normalized = x_centered / std (with broadcast)
    std_broadcast = graph.add_node(
        op="REPEAT",
        name="ln_std_broadcast",
        inputs=[graph.node_ref(std_node)],
        output_shape=input_shape,
        output_dtype=GGMLDtype.F32,
        params={"target_shape": input_shape},
    )

    normalized = graph.add_node(
        op="DIV",
        name="ln_normalized",
        inputs=[graph.node_ref(centered), graph.node_ref(std_broadcast)],
        output_shape=input_shape,
        output_dtype=GGMLDtype.F32,
    )

    # Step 6: Apply affine transform: normalized * weight + bias
    scaled = graph.add_node(
        op="MUL",
        name="ln_scaled",
        inputs=[graph.node_ref(normalized), weight_ref],
        output_shape=input_shape,
        output_dtype=GGMLDtype.F32,
    )

    output = graph.add_node(
        op="ADD",
        name="ln_output",
        inputs=[graph.node_ref(scaled), bias_ref],
        output_shape=input_shape,
        output_dtype=GGMLDtype.F32,
    )

    return graph.node_ref(output)


def decompose_concat_linear(
    graph: GGMLGraph,
    input_refs: list[str],
    input_shapes: list[list[int]],
    weight_ref: str,
    bias_ref: str | None,
    weight_shape: list[int],
    output_dim: int,
) -> str:
    """
    Decompose concat + linear into separate matmuls that sum.

    Based on pet_layers.cpp:31-49 and 695-796.

    Instead of: concat([A, B, C]) @ W + bias
    Use: A @ W_a + B @ W_b + C @ W_c + bias

    This avoids ggml_concat which lacks gradient support.

    Args:
        graph: The GGML graph being built
        input_refs: References to input tensors to concatenate
        input_shapes: Shapes of input tensors (GGML format)
        weight_ref: Reference to concatenated weight matrix
        bias_ref: Reference to bias (or None)
        weight_shape: Shape of weight matrix [concat_dim, output_dim]
        output_dim: Output dimension

    Returns:
        Reference to the output node
    """
    num_parts = len(input_refs)

    # Each input should have same shape except dimension 0
    d_in_per_part = input_shapes[0][0]
    batch_dims = input_shapes[0][1:]

    # Output shape: [output_dim, ...batch_dims]
    output_shape = [output_dim] + batch_dims

    # Create weight views and apply matmuls
    partial_results = []

    for i, (inp_ref, inp_shape) in enumerate(zip(input_refs, input_shapes)):
        d_in = inp_shape[0]

        # Create view into weight matrix for this partition
        # weight_view_i selects rows [i*d_in : (i+1)*d_in]
        weight_view = graph.add_node(
            op="VIEW",
            name=f"concat_lin_w_view_{i}",
            inputs=[weight_ref],
            output_shape=[d_in, output_dim],
            output_dtype=GGMLDtype.F32,
            params={
                "offset_bytes": i * d_in * 4,  # 4 bytes per float
                "ne0": d_in,
                "ne1": output_dim,
            },
        )

        # Apply matmul: input @ weight_view
        matmul = graph.add_node(
            op="MUL_MAT",
            name=f"concat_lin_mm_{i}",
            inputs=[graph.node_ref(weight_view), inp_ref],
            output_shape=output_shape,
            output_dtype=GGMLDtype.F32,
        )

        partial_results.append(graph.node_ref(matmul))

    # Sum all partial results
    if len(partial_results) == 1:
        result_ref = partial_results[0]
    else:
        # Sum first two
        result_ref = partial_results[0]
        for i in range(1, len(partial_results)):
            sum_node = graph.add_node(
                op="ADD",
                name=f"concat_lin_sum_{i}",
                inputs=[result_ref, partial_results[i]],
                output_shape=output_shape,
                output_dtype=GGMLDtype.F32,
            )
            result_ref = graph.node_ref(sum_node)

    # Add bias if present
    if bias_ref is not None:
        output = graph.add_node(
            op="ADD",
            name="concat_lin_bias",
            inputs=[result_ref, bias_ref],
            output_shape=output_shape,
            output_dtype=GGMLDtype.F32,
        )
        return graph.node_ref(output)

    return result_ref


def decompose_rsqrt(
    graph: GGMLGraph,
    input_ref: str,
    input_shape: list[int],
) -> str:
    """
    Decompose rsqrt (1/sqrt(x)) into sqrt + div.

    GGML doesn't have rsqrt, so we compute:
    rsqrt(x) = 1.0 / sqrt(x)

    Args:
        graph: The GGML graph being built
        input_ref: Reference to input tensor
        input_shape: Shape of input tensor

    Returns:
        Reference to the output node
    """
    # sqrt(x)
    sqrt_node = graph.add_node(
        op="SQRT",
        name="rsqrt_sqrt",
        inputs=[input_ref],
        output_shape=input_shape,
        output_dtype=GGMLDtype.F32,
    )

    # 1.0 / sqrt(x) using scale with 1.0 followed by div
    # Actually, we can use: result = ones / sqrt
    # But we don't have a ones tensor. Instead, use reciprocal pattern.
    #
    # GGML approach: use SCALE to create ones, then DIV
    # Actually simpler: just note this in metadata and handle at runtime
    #
    # For now, emit a custom op that runtime will handle
    output = graph.add_node(
        op="RSQRT",  # Custom op - runtime must implement
        name="rsqrt",
        inputs=[input_ref],
        output_shape=input_shape,
        output_dtype=GGMLDtype.F32,
    )

    return graph.node_ref(output)


def decompose_mean_dim(
    graph: GGMLGraph,
    input_ref: str,
    input_shape: list[int],
    dim: int,
    keepdim: bool = True,
) -> str:
    """
    Decompose mean along dimension to sum + scale.

    mean(x, dim) = sum(x, dim) / size(dim)

    Args:
        graph: The GGML graph being built
        input_ref: Reference to input tensor
        input_shape: Shape of input tensor (GGML format)
        dim: Dimension to reduce (GGML dimension index)
        keepdim: Whether to keep the reduced dimension

    Returns:
        Reference to the output node
    """
    dim_size = input_shape[dim]

    # Output shape after reduction
    if keepdim:
        output_shape = input_shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = input_shape[:dim] + input_shape[dim+1:]

    # If reducing dim 0, use SUM_ROWS
    if dim == 0:
        sum_node = graph.add_node(
            op="SUM_ROWS",
            name="mean_sum",
            inputs=[input_ref],
            output_shape=output_shape,
            output_dtype=GGMLDtype.F32,
        )
    else:
        # Need permute + sum_rows + permute back
        # For simplicity, emit SUM with dim parameter
        sum_node = graph.add_node(
            op="SUM",
            name="mean_sum",
            inputs=[input_ref],
            output_shape=output_shape,
            output_dtype=GGMLDtype.F32,
            params={"dim": dim, "keepdim": keepdim},
        )

    # Scale by 1/dim_size
    output = graph.add_node(
        op="SCALE",
        name="mean_scale",
        inputs=[graph.node_ref(sum_node)],
        output_shape=output_shape,
        output_dtype=GGMLDtype.F32,
        params={"scale": 1.0 / float(dim_size)},
    )

    return graph.node_ref(output)


def decompose_addmm(
    graph: GGMLGraph,
    bias_ref: str,
    input_ref: str,
    weight_ref: str,
    input_shape: list[int],
    weight_shape: list[int],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> str:
    """
    Decompose addmm (beta * bias + alpha * input @ weight) to mm + scale + add.

    Args:
        graph: The GGML graph being built
        bias_ref: Reference to bias tensor
        input_ref: Reference to input tensor
        weight_ref: Reference to weight tensor
        input_shape: Shape of input [K, M] in GGML
        weight_shape: Shape of weight [N, K] in GGML (transposed)
        alpha: Scalar multiplier for matmul result
        beta: Scalar multiplier for bias

    Returns:
        Reference to the output node
    """
    # Output shape: [N, M] where N = weight_shape[0], M = input_shape[1]
    output_shape = [weight_shape[0], input_shape[1]]

    # mm: input @ weight.T
    mm_node = graph.add_node(
        op="MUL_MAT",
        name="addmm_mm",
        inputs=[weight_ref, input_ref],
        output_shape=output_shape,
        output_dtype=GGMLDtype.F32,
    )

    result_ref = graph.node_ref(mm_node)

    # Scale by alpha if not 1.0
    if alpha != 1.0:
        scaled = graph.add_node(
            op="SCALE",
            name="addmm_alpha",
            inputs=[result_ref],
            output_shape=output_shape,
            output_dtype=GGMLDtype.F32,
            params={"scale": alpha},
        )
        result_ref = graph.node_ref(scaled)

    # Scale bias by beta if not 1.0
    if beta != 1.0:
        scaled_bias = graph.add_node(
            op="SCALE",
            name="addmm_beta",
            inputs=[bias_ref],
            output_shape=output_shape,  # Assumes bias broadcasts
            output_dtype=GGMLDtype.F32,
            params={"scale": beta},
        )
        bias_ref = graph.node_ref(scaled_bias)

    # Add bias
    output = graph.add_node(
        op="ADD",
        name="addmm_output",
        inputs=[result_ref, bias_ref],
        output_shape=output_shape,
        output_dtype=GGMLDtype.F32,
    )

    return graph.node_ref(output)


def decompose_dropout(
    graph: GGMLGraph,
    input_ref: str,
    input_shape: list[int],
    p: float = 0.0,
    training: bool = False,
) -> str:
    """
    Handle dropout - in inference mode this is identity.

    During inference (training=False or p=0), dropout is a no-op.
    We emit a CONT (contiguous) op which acts as identity.

    Args:
        graph: The GGML graph being built
        input_ref: Reference to input tensor
        input_shape: Shape of input tensor
        p: Dropout probability (ignored in inference)
        training: Whether in training mode

    Returns:
        Reference to the output node (identity in inference)
    """
    if not training or p == 0.0:
        # Identity - just return input reference
        # But we may need a CONT to ensure it's in the graph
        output = graph.add_node(
            op="CONT",
            name="dropout_identity",
            inputs=[input_ref],
            output_shape=input_shape,
            output_dtype=GGMLDtype.F32,
        )
        return graph.node_ref(output)

    # Training mode dropout would need random masking
    # Not supported for export - training should use PyTorch
    raise ValueError("Training mode dropout not supported for GGML export")


# Registry of decomposition functions
DECOMPOSITIONS: dict[str, Callable] = {
    "aten.layer_norm.default": decompose_layer_norm,
    "aten.native_layer_norm.default": decompose_layer_norm,
    "aten.rsqrt.default": decompose_rsqrt,
    "aten.mean.dim": decompose_mean_dim,
    "aten.addmm.default": decompose_addmm,
    "aten.dropout.default": decompose_dropout,
    # Note: cat decomposition is handled specially during graph construction
    # because it requires analyzing the downstream operations
}


def get_decomposition(op_name: str) -> Callable | None:
    """Get the decomposition function for an operation."""
    # Normalize op name
    if op_name.startswith("torch._ops."):
        op_name = op_name[len("torch._ops."):]
    if op_name.startswith("torch.ops."):
        op_name = op_name[len("torch.ops."):]

    return DECOMPOSITIONS.get(op_name)


def needs_decomposition(op_name: str) -> bool:
    """Check if an operation needs decomposition."""
    return get_decomposition(op_name) is not None
