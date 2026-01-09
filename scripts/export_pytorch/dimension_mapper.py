"""
Dimension mapping between PyTorch and GGML.

GGML uses reversed dimension ordering from PyTorch:
- PyTorch: [N, C, H, W] (batch, channel, height, width)
- GGML: [W, H, C, N] (ne[0], ne[1], ne[2], ne[3])

This module provides utilities for converting shapes and dimension indices.
"""

from __future__ import annotations


def pytorch_to_ggml_shape(shape: list[int] | tuple[int, ...]) -> list[int]:
    """
    Convert PyTorch shape to GGML shape (reverse order).

    Examples:
        >>> pytorch_to_ggml_shape([8, 7, 256])  # [batch, seq, features]
        [256, 7, 8]
        >>> pytorch_to_ggml_shape([32, 64])  # [batch, features]
        [64, 32]

    Args:
        shape: PyTorch tensor shape

    Returns:
        GGML tensor shape (reversed)
    """
    return list(reversed(shape))


def ggml_to_pytorch_shape(shape: list[int] | tuple[int, ...]) -> list[int]:
    """
    Convert GGML shape to PyTorch shape (reverse order).

    Examples:
        >>> ggml_to_pytorch_shape([256, 7, 8])  # [features, seq, batch]
        [8, 7, 256]

    Args:
        shape: GGML tensor shape (ne[0], ne[1], ...)

    Returns:
        PyTorch tensor shape
    """
    return list(reversed(shape))


def pytorch_to_ggml_dim(dim: int, ndim: int) -> int:
    """
    Convert a PyTorch dimension index to GGML dimension index.

    In PyTorch, dim 0 is the outermost (batch) dimension.
    In GGML, dim 0 (ne[0]) is the innermost (contiguous) dimension.

    Examples:
        >>> pytorch_to_ggml_dim(0, 3)  # batch dim in 3D tensor
        2
        >>> pytorch_to_ggml_dim(2, 3)  # innermost dim in 3D tensor
        0
        >>> pytorch_to_ggml_dim(-1, 3)  # last dim (feature dim)
        0

    Args:
        dim: PyTorch dimension index (can be negative)
        ndim: Number of dimensions in the tensor

    Returns:
        GGML dimension index
    """
    # Handle negative dimensions
    if dim < 0:
        dim = ndim + dim
    # Reverse the dimension index
    return ndim - 1 - dim


def ggml_to_pytorch_dim(dim: int, ndim: int) -> int:
    """
    Convert a GGML dimension index to PyTorch dimension index.

    Args:
        dim: GGML dimension index (ne[dim])
        ndim: Number of dimensions in the tensor

    Returns:
        PyTorch dimension index
    """
    return ndim - 1 - dim


def pytorch_to_ggml_permute(perm: list[int] | tuple[int, ...], ndim: int) -> list[int]:
    """
    Convert PyTorch permute dimensions to GGML permute dimensions.

    In PyTorch: permute([0, 2, 1, 3]) on shape [a, b, c, d] -> [a, c, b, d]
    In GGML: same logical operation needs adjusted indices

    Examples:
        >>> pytorch_to_ggml_permute([0, 2, 1], 3)  # Swap last two dims
        [0, 2, 1]  # Same in GGML but operates on reversed shape
        >>> pytorch_to_ggml_permute([1, 0], 2)  # Transpose 2D
        [1, 0]

    Args:
        perm: PyTorch permutation (output dim i gets input dim perm[i])
        ndim: Number of dimensions

    Returns:
        GGML permutation
    """
    # For a permutation that takes PyTorch dims and rearranges them,
    # we need to map it to GGML's reversed dimension space.
    #
    # If PyTorch permute is [p0, p1, p2, p3] meaning:
    #   output[i] = input[perm[i]]
    #
    # In GGML (reversed), the equivalent permute operates on ne[] indices.
    # GGML ne[i] corresponds to PyTorch shape[ndim-1-i]
    #
    # The GGML permute needs to be: for each GGML output dim j,
    # which GGML input dim does it come from?

    # Map PyTorch dims to GGML dims
    ggml_perm = []
    for pt_out_dim in range(ndim):
        pt_in_dim = perm[pt_out_dim]
        # Convert both to GGML space
        ggml_out_dim = pytorch_to_ggml_dim(pt_out_dim, ndim)
        ggml_in_dim = pytorch_to_ggml_dim(pt_in_dim, ndim)
        ggml_perm.append((ggml_out_dim, ggml_in_dim))

    # Sort by output dim and extract input dims
    ggml_perm.sort(key=lambda x: x[0])
    return [x[1] for x in ggml_perm]


def pytorch_to_ggml_transpose_dims(dim0: int, dim1: int, ndim: int) -> tuple[int, int]:
    """
    Convert PyTorch transpose dimensions to GGML.

    Args:
        dim0: First PyTorch dimension
        dim1: Second PyTorch dimension
        ndim: Number of dimensions

    Returns:
        Tuple of (ggml_dim0, ggml_dim1)
    """
    return (
        pytorch_to_ggml_dim(dim0, ndim),
        pytorch_to_ggml_dim(dim1, ndim),
    )


def make_ggml_view_params(
    original_shape: list[int],
    view_shape: list[int],
    offset: int = 0,
) -> dict:
    """
    Calculate GGML view parameters from PyTorch shapes.

    Args:
        original_shape: PyTorch shape of source tensor
        view_shape: PyTorch shape of view
        offset: Byte offset into source tensor

    Returns:
        Dict with GGML view parameters (ne0, ne1, ..., nb1, nb2, ..., offset)
    """
    ggml_shape = pytorch_to_ggml_shape(view_shape)
    ggml_orig = pytorch_to_ggml_shape(original_shape)

    # Calculate strides (in elements, not bytes)
    # GGML stride for dim i is product of all dims j < i
    strides = [1]
    for i in range(len(ggml_orig) - 1):
        strides.append(strides[-1] * ggml_orig[i])

    params = {
        "shape": ggml_shape,
        "offset": offset,
    }

    # Add strides for dimensions > 0
    for i, stride in enumerate(strides[1:], start=1):
        params[f"nb{i}"] = stride

    return params


def calculate_broadcast_shape(shape1: list[int], shape2: list[int]) -> list[int]:
    """
    Calculate the broadcast result shape for two tensors.

    Uses NumPy/PyTorch broadcasting rules.

    Args:
        shape1: First tensor shape (PyTorch ordering)
        shape2: Second tensor shape (PyTorch ordering)

    Returns:
        Broadcast result shape
    """
    # Pad shorter shape with 1s on the left
    max_len = max(len(shape1), len(shape2))
    shape1 = [1] * (max_len - len(shape1)) + list(shape1)
    shape2 = [1] * (max_len - len(shape2)) + list(shape2)

    result = []
    for s1, s2 in zip(shape1, shape2):
        if s1 == s2:
            result.append(s1)
        elif s1 == 1:
            result.append(s2)
        elif s2 == 1:
            result.append(s1)
        else:
            raise ValueError(f"Cannot broadcast shapes {shape1} and {shape2}")

    return result


def needs_contiguous(op: str) -> bool:
    """
    Check if an operation requires contiguous input tensors.

    In GGML, operations like MUL_MAT require contiguous tensors.
    After permute/transpose, ggml_cont() must be called.

    Args:
        op: GGML operation name

    Returns:
        True if the operation requires contiguous inputs
    """
    # Operations that require contiguous tensors
    contiguous_ops = {
        "MUL_MAT",
        "SOFT_MAX",
        "FLASH_ATTN_EXT",
        "CONV_1D",
        "CONV_2D",
        "POOL_1D",
        "POOL_2D",
    }
    return op in contiguous_ops
