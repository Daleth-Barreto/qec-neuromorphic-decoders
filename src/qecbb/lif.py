"""Neurona LIF (idéntica a ConvSNNDecoderFlow)."""
import torch

class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        mask = (input > -1.0) & (input < 1.0)
        return grad_input * mask.float()

spike_fn = SurrogateHeaviside.apply
LIF_THRESHOLD = 1.0