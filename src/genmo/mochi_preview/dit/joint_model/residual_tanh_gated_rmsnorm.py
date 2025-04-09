import torch
import ipdb

class ResidualTanhGatedRMSNorm(torch.nn.Module):
    def __init__(self):
        super(ResidualTanhGatedRMSNorm, self).__init__()

    def forward(self, x, x_res, gate, eps=1e-6):
        # Convert to fp32 for precision
        x_res_fp32 = x_res.float()

        # Compute RMS
        mean_square = x_res_fp32.pow(2).mean(-1, keepdim=True)
        scale = torch.rsqrt(mean_square + eps)

        tanh_gate = torch.tanh(gate).unsqueeze(1)

        # # Normalize and apply gated scaling643
        x_normed = x_res_fp32 * scale * tanh_gate

        # Apply residual connection
        output = x + x_normed.type_as(x)

        return output

def residual_tanh_gated_rmsnorm(x, x_res, gate, eps=1e-6):
    norm = ResidualTanhGatedRMSNorm()
    return norm.forward(x, x_res, gate, eps)
