# All code and comments in English only.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


def trunc_normal_(w, std=0.02):
    nn.init.trunc_normal_(w, std=std)


class MambaInMamba(nn.Module):
    """
    Mamba-in-Mamba: use a small Mamba to compute delta (dt) instead of a linear projection.
    This version adds stability guards to avoid NaNs.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        # Inner Mamba size
        dt_mamba_d_state: int = 4,
        dt_mamba_d_conv: int = 2,
        device=None,
        dtype=None,
        # Extra stabilizers
        clamp_delta: float = 6.0,   # clamp pre-softplus delta to [-clamp_delta, clamp_delta]
        init_std: float = 0.02,     # small init for projections
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.clamp_delta = clamp_delta

        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        trunc_normal_(self.in_proj.weight, std=init_std)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)

        # Depthwise conv
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # x projection -> [dt_rank | B | C]
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        trunc_normal_(self.x_proj.weight, std=init_std)

        # Inner Mamba for delta
        self.dt_mamba = Mamba(
            d_model=self.dt_rank,
            d_state=dt_mamba_d_state,
            d_conv=dt_mamba_d_conv,
            expand=1,
            dt_rank="auto",
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=True,
            bias=False,
            use_fast_path=False,   # stability first
            layer_idx=None,
            **factory_kwargs
        )

        # Out proj from inner Mamba to d_inner
        self.dt_out_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        nn.init.zeros_(self.dt_out_proj.weight)  # bias-only start for stable dt

        # Initialize dt_out_proj.bias as in original Mamba
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_out_proj.bias.copy_(inv_dt)
        self.dt_out_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D skip
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        trunc_normal_(self.out_proj.weight, std=init_std)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        # For visualization
        self.last_delta = None

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        returns: (B, L, D)
        """
        Bsz, L, _ = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, Bsz)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # Input projection
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=L,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)

        # Conv
        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :L])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # x -> [dt_in | B | C]
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt_in, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # inner Mamba over (B, L, dt_rank)
        dt_in = rearrange(dt_in, "(b l) d -> b l d", b=Bsz, l=L)
        dt_mamba_out = self.dt_mamba(dt_in)                # (B, L, dt_rank)
        dt_mamba_out = rearrange(dt_mamba_out, "b l d -> (b l) d")

        # project to d_inner then clamp before softplus inside selective_scan
        delta = self.dt_out_proj(dt_mamba_out)             # (B*L, d_inner)
        delta = torch.clamp(delta, -self.clamp_delta, self.clamp_delta)
        delta = rearrange(delta, "(b l) d -> b d l", b=Bsz, l=L)

        # store for visualization
        self.last_delta = delta.detach()

        # B, C to scan shapes
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

        # selective scan (dt_softplus=True applies softplus internally)
        y = selective_scan_fn(
            x,
            delta,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_out_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )

        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)

        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        """
        Single-step decode. Keep logic consistent with forward.
        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only 1 token at a time"

        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)
        dt_in, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # inner Mamba for 1-step
        dt_in = dt_in.unsqueeze(1)              # (B, 1, dt_rank)
        dt_mamba_out = self.dt_mamba(dt_in)     # (B, 1, dt_rank)
        dt = self.dt_out_proj(dt_mamba_out.squeeze(1))  # (B, d_inner)
        dt = torch.clamp(dt, -self.clamp_delta, self.clamp_delta)

        A = -torch.exp(self.A_log.float())

        # State update; let kernel softplus dt
        if selective_state_update is None:
            dt = F.softplus(dt + self.dt_out_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + x.unsqueeze(-1) * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z,
                dt_bias=self.dt_out_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_out_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_out_proj.weight.device,
                dtype=self.dt_out_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
