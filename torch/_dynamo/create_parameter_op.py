import torch

import logging

torch_log = logging.getLogger("torch")

doc = """
This is used when dynamo traces torch.nn.Parameter, which normally would not trace properly
with AOTAutograd.  We instead create a placeholder torch.nn.Parameter before the graph, which
becomes a graph arg and has no storage backing it.  At the point in the graph where the parameter
actually should be created we mutate this sacrificial placeholder into it.  This allows gradients
to flow into the parameter as if it were an input to the graph (which is the only thing we are
allowed to compute gradients on).
""".strip()


class TracableCreateParameter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, placeholder):
        assert not tensor.requires_grad
        return placeholder.set_(tensor)
        # NOTE(yf225): set_ doesn't work if placeholder is Tensor but `tensor` is DTensor. So we use copy_ instead and rely on compiler to elide the copy.
        # This shows up in compiled FSDP+TP.
        # return placeholder.copy_(tensor)

    @staticmethod
    def backward(ctx, grad):
        return None, grad  # grad flows to placeholder


def tracable_create_parameter(tensor, placeholder):
    with torch.set_grad_enabled(placeholder.requires_grad):
        out = TracableCreateParameter.apply(tensor, placeholder)
        out = out.clone()
    return out


# def new_parameter_placeholder(data_tensor, requires_grad):
#     """Create a placeholder to be passed to the above functions"""
#     new_data_tensor = torch.empty_like(data_tensor)
#     torch_log.warning(f"new_data_tensor: {new_data_tensor}")
#     result = torch.nn.Parameter(
#         new_data_tensor, requires_grad=requires_grad
#     )
#     # NOTE(yf225): resize_ doesn't work for DTensor or any other tensor subclass. Do we really need it? (shouldn't result just be a FakeTensor and no real storage?)
#     # # TODO(jansel): alloc followed by free is inefficient, need a way to allocate an unbacked tensor.
#     # # Allocating a zero tensor would causes assert failures in autograd.
#     # result.untyped_storage().resize_(0)
#     torch_log.warning(f"result: {result}")
#     return result


def new_parameter_placeholder(size, dtype, device, requires_grad, is_dtensor):
    """Create a placeholder to be passed to the above functions"""
    data_tensor = torch.empty(size, dtype=dtype, device=device)
    data_tensor.untyped_storage().resize_(0)
    if is_dtensor:
        data_tensor = torch.distributed._tensor.api.DTensor.from_local(data_tensor)
    result = torch.nn.Parameter(
        data_tensor, requires_grad=requires_grad
    )
    # TODO(jansel): alloc followed by free is inefficient, need a way to allocate an unbacked tensor.
    # Allocating a zero tensor would causes assert failures in autograd.
    # result.untyped_storage().resize_(0)
    return result
