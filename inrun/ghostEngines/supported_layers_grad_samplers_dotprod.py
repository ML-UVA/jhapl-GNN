from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.dense.linear import Linear as PyGLinear
import torch
#import transformers.pytorch_utils
from torch import nn
import torch.nn.functional as F


def _should_use_ghost_computation(layer: nn.Module, A: torch.Tensor, B: torch.Tensor, conv: bool = False):
    """
    Determines whether to use the efficient "ghost" computation method based
    on the dimensions of the activation and backpropagation tensors.

    This check is based on the heuristic described in the literature, comparing
    the computational cost of materializing the full gradient versus using the
    ghost computation trick.

    Args:
        layer: The neural network layer.
        A: The activation tensor.
        B: The backpropagation tensor.
        conv: Flag indicating if the layer is a convolutional layer.
    """
    # The check only needs to be performed once per layer.
    if hasattr(layer, "use_ghost_computation"):
        return

    if not conv:
        # For linear layers
        T = torch.prod(torch.tensor(A.shape[1:-1])).item() if A.dim() > 2 else 1
        d = A.shape[-1]
        p = B.shape[-1]
    else:
        # For convolutional layers (after unfolding)
        T = A.shape[-1]
        d = A.shape[1]
        p = B.shape[1]

    # The total number of parameters in the weight matrix
    #num_weight_params = layer.weight.numel()
    weight = layer.lin.weight if hasattr(layer, 'lin') else layer.weight
    num_weight_params = weight.numel()

    # Test: 2*T^2 <= d*p  (or num_weight_params)
    layer.use_ghost_computation = bool(2 * T**2 <= num_weight_params)


def _create_or_accumulate_train_grad(param: torch.Tensor, grad: torch.Tensor) -> None:
    """Creates or accumulates a gradient for a given parameter in the .train_grad attribute.

    This function adds a computed gradient to the .train_grad attribute of a parameter.
    It handles both the initial creation of the .train_grad attribute and the
    subsequent accumulation of gradients, preventing contamination of the standard
    .grad attribute.

    Args:
        param: The model parameter to which the training gradient will be added.
        grad: The newly computed training gradient.
    """
    # Ensure the new gradient has the same shape as the parameter.
    assert grad.shape == param.shape, \
        f"Gradient shape ({grad.shape}) does not match parameter shape ({param.shape})"

    # Detach the gradient to ensure it's not part of any further computation graph.
    new_grad = grad.detach()

    # Check if the parameter already has a train_grad attribute.
    if hasattr(param, 'train_grad'):
        # Add the new gradient to the existing one in-place.
        param.train_grad.add_(new_grad)
    else:
        # If the parameter does not have a train_grad attribute yet, create it.
        param.train_grad = new_grad


# Linear Layer Implementation
# #############################################################################

def _compute_linear_dot_product(layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, val_batch_size: int):
    """Computes the gradient dot-product for an nn.Linear layer.

    Supports both 2D inputs (batch, d) and higher-rank inputs
    (batch, ..., d) such as language models with a sequence dimension.
    """

    if A is None:
        raise ValueError("Input activations A cannot be None.")
    if B is None:
        raise ValueError("Output gradients B cannot be None.")

    # Detach to keep them out of autograd graph for efficiency/safety
    A = A.detach()
    B = B.detach()

    # Cast to bfloat16 for the dot-product path; train-grad path uses original types
    A_bf16 = A.to(torch.bfloat16)
    B_bf16 = B.to(torch.bfloat16)

    total_bs = A.size(0)
    train_bs = total_bs - val_batch_size
    if train_bs <= 0:
        raise ValueError("No training samples to compute dot product, check batch sizes.")

    # Decide whether to use ghost computation
    _should_use_ghost_computation(layer, A, B)

    # Common split for bias computation and non-ghost fallback
    A_train_full, A_val_full = torch.split(A, [train_bs, val_batch_size], dim=0)
    B_train_full, B_val_full = torch.split(B, [train_bs, val_batch_size], dim=0)

    # --- Weight dot product ---
    if layer.use_ghost_computation:
        # Flatten any non-batch, non-feature dims into a single token dimension
        d = A_bf16.size(-1)
        p = B_bf16.size(-1)

        # Reshape to [batch, T, feat]
        T = int(A_bf16.numel() // (total_bs * d))
        A_bf16 = A_bf16.reshape(total_bs, T, d)
        B_bf16 = B_bf16.reshape(total_bs, T, p)

        # Split train/val
        A_train = A_bf16[:train_bs].reshape(-1, d).contiguous()  # [(train_bs·T), d]
        A_val = A_bf16[train_bs:].reshape(-1, d).contiguous()    # [(val_bs  ·T), d]
        B_train = B_bf16[:train_bs].reshape(-1, p).contiguous()  # [(train_bs·T), p]
        B_val = B_bf16[train_bs:].reshape(-1, p).contiguous()    # [(val_bs  ·T), p]

        # Two GEMMs → (train_tokens × val_tokens)
        a_dot = torch.matmul(A_train, A_val.T)
        b_dot = torch.matmul(B_train, B_val.T)

        # Hadamard + reduce over validation tokens; fold back token dimension
        token_contrib = (a_dot * b_dot).sum(dim=1, dtype=torch.float32)
        layer.weight.grad_dot_prod = token_contrib.reshape(train_bs, T).sum(dim=1)
    else:
        # Materialize grads and compute Frobenius inner product
        grad_train = torch.einsum('b...d,b...p->bpd', A_train_full, B_train_full)
        grad_val = torch.einsum('...d,...p->dp', A_val_full.sum(dim=0), B_val_full.sum(dim=0))
        layer.weight.grad_dot_prod = torch.einsum('dp,bpd->b', grad_val, grad_train)

    # --- Bias dot product ---
    if layer.bias is not None:
        # Use original dtype tensors for bias path
        B_train = B_train_full
        B_val = B_val_full
        sum_dims_val = list(range(B_val.dim() - 1))
        grad_bias_val = B_val.sum(dim=sum_dims_val)
        sum_dims_train = list(range(1, B_train.dim() - 1))
        grad_bias_train = B_train.sum(dim=sum_dims_train) if B_train.dim() > 2 else B_train
        layer.bias.grad_dot_prod = torch.einsum('p,bp->b', grad_bias_val, grad_bias_train)

    # torch.cuda.synchronize()  # Ensure all operations are complete
    # end_time = time.time()
    # print(f"Debug: Dot product computation time for lm_head: {(end_time - start_time) * 1000:.4f} ms")
    # print(f"Debug: Check grad dot product value for Linear layer: {layer.weight.grad_dot_prod}")



def _compute_linear_train_grad(layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, val_batch_size: int):
    """
    Computes the training gradient for an nn.Linear layer's weight.
    This version always computes the average gradient to match PyTorch's default behavior.
    """
    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        return None # Return None if there's nothing to compute

    A_train, _ = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    # Compute the SUM of gradients over the training batch
    grad_weight = torch.einsum('b...p,b...d->pd', B_train, A_train)

    # Always divide by the number of training samples to get the AVERAGE gradient
    grad_weight /= train_batch_size
    
    return grad_weight

# LayerNorm Layer Implementation
# #############################################################################

def _compute_layernorm_dot_product(layer: nn.LayerNorm, A: torch.Tensor, B: torch.Tensor, val_batch_size: int):
    """Computes the gradient dot-product for an nn.LayerNorm layer."""

    # Detach the tensors to ensure they are not part of the computation graph.
    A = A.detach()
    B = B.detach()

    # Cast A and B to bfloat16 for efficiency.
    A = A.to(torch.bfloat16)
    B = B.to(torch.bfloat16)

    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        return

    A_train, A_val = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, B_val = torch.split(B, [train_batch_size, val_batch_size], dim=0)
    
    # --- Weight dot product ---
    # The gradient for the weight is B * normalized_A
    normalized_A_train = F.layer_norm(A_train, layer.normalized_shape, eps=layer.eps)
    normalized_A_val = F.layer_norm(A_val, layer.normalized_shape, eps=layer.eps)

    grad_weight_train = B_train * normalized_A_train
    grad_weight_val = B_val * normalized_A_val

    # Reduce training per-sample gradients over non-feature dims only → [B_train, F]
    # Keep the last dim (feature) intact for a correct dot with validation vector.
    if grad_weight_train.dim() >= 2:
        sum_dims_train = list(range(1, grad_weight_train.dim() - 1))
        per_sample_grad_weight = grad_weight_train.sum(dim=sum_dims_train) if sum_dims_train else grad_weight_train
    else:
        per_sample_grad_weight = grad_weight_train

    # Aggregate validation gradient over batch and token dims only → [F]
    sum_dims_val = list(range(grad_weight_val.dim() - 1))
    total_grad_weight_val = grad_weight_val.sum(dim=sum_dims_val)

    # Feature-wise inner product to obtain per-sample scalars → [B_train]
    layer.weight.grad_dot_prod = torch.einsum(
        'bf,f->b', per_sample_grad_weight.float(), total_grad_weight_val.float()
    )

    # --- Bias dot product ---
    if layer.bias is not None:
        # Bias gradient is B; reduce non-feature dims for train → [B_train, F]
        if B_train.dim() >= 2:
            sum_dims_train = list(range(1, B_train.dim() - 1))
            per_sample_grad_bias = B_train.sum(dim=sum_dims_train) if sum_dims_train else B_train
        else:
            per_sample_grad_bias = B_train

        # Validation aggregate over batch and token dims only → [F]
        sum_dims_val = list(range(B_val.dim() - 1))
        total_grad_bias_val = B_val.sum(dim=sum_dims_val)

        layer.bias.grad_dot_prod = torch.einsum(
            'bf,f->b', per_sample_grad_bias.float(), total_grad_bias_val.float()
        )


def _compute_layernorm_train_grad(
    layer: nn.LayerNorm,
    A: torch.Tensor,
    B: torch.Tensor,
    val_batch_size: int
) -> None:
    """
    Computes and directly applies the training gradient for an nn.LayerNorm layer.

    --- Design Reasoning ---
    This function is handled differently from `_compute_linear_train_grad` because
    LayerNorm's gradient calculations are self-contained. The gradients for both
    its parameters (weight and bias) are computed from the same inputs (A and B).

    - Weight Gradient: Depends on both the input activations `A` (specifically, the
      normalized version of `A`) and the backpropagated gradients `B`.
    - Bias Gradient: Depends only on the backpropagated gradients `B`.

    Because both computations use the same `A` and `B` tensors and are closely
    related, it is cleaner and more efficient to calculate and apply them within
    this single function. This approach avoids passing intermediate results (like
    `normalized_A`) out to a dispatcher.

    Therefore, this function computes and applies the gradients for both weight
    and bias internally and returns `None` to signal to the dispatcher that its
    work for this layer is complete.
    """

    train_batch_size = A.size(0) - val_batch_size
    if train_batch_size <= 0:
        raise ValueError("No training samples to compute gradients, check batch sizes.")
    
    # debug: print out the shapes of A and B & the first few elements
    # print(f"[Grad] Layer Name: {layer.__class__.__name__}")
    # print(f"[Grad] A shape: {A.shape}, B shape: {B.shape}, val_batch_size: {val_batch_size}")
    # print(f"[Grad] A (first 5): {A[:5]}")
    # print(f"B (first 5): {B[:5]}")

    A_train, _ = torch.split(A, [train_batch_size, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_batch_size, val_batch_size], dim=0)

    # --- Weight (gamma) gradient ---
    # The gradient is B * normalized_A. We sum over all dimensions except the
    # final feature dimension to match the shape of the weight parameter.
    normalized_A_train = F.layer_norm(A_train, layer.normalized_shape, eps=layer.eps)
    grad_weight = (B_train * normalized_A_train).sum(dim=list(range(A_train.dim() - 1)))

    grad_weight /= train_batch_size

    # Apply the computed weight gradient directly.
    _create_or_accumulate_train_grad(layer.weight, grad_weight)

    # --- Bias (beta) gradient ---
    if layer.bias is not None:
        # The gradient is just B, summed over all dimensions except features.
        grad_bias = B_train.sum(dim=list(range(B_train.dim() - 1)))

        grad_bias /= train_batch_size

        # Apply the computed bias gradient directly.
        _create_or_accumulate_train_grad(layer.bias, grad_bias)

    # Return None because this function handles its own gradient application.
    return None


def _compute_gcnconv_dot_product(layer, A, B, val_batch_size):
    """
    A: aggregated node features after adjacency  [N, F]
    B: backprop gradients at output              [N, P]
    """
    if A is None:
        raise ValueError("Input activations A cannot be None.")
    if B is None:
        raise ValueError("Output gradients B cannot be None.")


    A = A.detach()
    B = B.detach()

    A_bf16 = A.to(torch.bfloat16)
    B_bf16 = B.to(torch.bfloat16)
    
    total_bs = A.size(0)
    train_bs = total_bs - val_batch_size
    if train_bs <= 0:
        raise ValueError("No training nodes")
    _should_use_ghost_computation(layer, A, B)
    
    A_train, A_val = torch.split(A, [train_bs, val_batch_size], dim=0)
    B_train, B_val = torch.split(B, [train_bs, val_batch_size], dim=0)

    #validation gradients
    #grad_val = torch.einsum("nf,np->fp", A_val.sum(dim=0), B_val.sum(dim=0))

    #training gradients
    #grad_train = torch.einsum("nf,np->nfp", A_train, B_train)

    # Frobenius inner product per node
    #layer.lin.weight.grad_dot_prod = torch.einsum("fp,nfp->n", grad_val, grad_train)
    grad_val   = torch.einsum("nf,np->fp", A_val, B_val)
    grad_train = torch.einsum("nf,np->nfp", A_train, B_train)
    layer.lin.weight.grad_dot_prod = torch.einsum("fp,nfp->n", grad_val, grad_train)

def _compute_gcnconv_train_grad(layer, A, B, val_batch_size):
    train_bs = A.size(0) - val_batch_size
    if train_bs <= 0:
        return None

    A_train, _ = torch.split(A, [train_bs, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_bs, val_batch_size], dim=0)

    grad_weight = torch.einsum("nf,np->fp", A_train, B_train)
    grad_weight /= train_bs

    return grad_weight
def _compute_gatconv_dot_product(layer, A, B, val_batch_size):

    A = A.detach()
    B = B.detach()

    total_bs = A.size(0)
    train_bs = total_bs - val_batch_size
    if train_bs <= 0:
        raise ValueError("No training nodes")

    A_train, A_val = torch.split(A, [train_bs, val_batch_size], dim=0)
    B_train, B_val = torch.split(B, [train_bs, val_batch_size], dim=0)

    grad_val = torch.einsum("nf,np->fp", A_val.sum(dim=0), B_val.sum(dim=0))
    grad_train = torch.einsum("nf,np->nfp", A_train, B_train)

    layer.lin.weight.grad_dot_prod = torch.einsum(
        "fp,nfp->n", grad_val, grad_train
    )
def _compute_gatconv_train_grad(layer, A, B, val_batch_size):
    train_bs = A.size(0) - val_batch_size
    if train_bs <= 0:
        return None

    A_train, _ = torch.split(A, [train_bs, val_batch_size], dim=0)
    B_train, _ = torch.split(B, [train_bs, val_batch_size], dim=0)

    grad_weight = torch.einsum("nf,np->fp", A_train, B_train)
    grad_weight /= train_bs

    return grad_weight


_supported_layers_dotprod = {
    nn.Linear: (_compute_linear_dot_product, _compute_linear_train_grad),
    nn.LayerNorm: (_compute_layernorm_dot_product, _compute_layernorm_train_grad),
    GCNConv: (_compute_gcnconv_dot_product, _compute_gcnconv_train_grad),
    GATConv: (_compute_gatconv_dot_product, _compute_gatconv_train_grad),
    PyGLinear: (_compute_linear_dot_product, _compute_linear_train_grad),
}
