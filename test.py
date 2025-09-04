# Copyright (c) 2023, Tri Dao, Albert Gu.
# Stabilized Mamba-in-Mamba block and CIFAR-10 grayscale trainer
# All comments in English only.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
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
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
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
        dt_mamba_d_state=4,
        dt_mamba_d_conv=2,
        device=None,
        dtype=None,
        # Extra stabilizers
        clamp_delta=6.0,       # clamp pre-softplus delta to [-clamp_delta, clamp_delta]
        init_std=0.02,         # small init for projections
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
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
            use_fast_path=False,   # disable fast path for stability
            layer_idx=None,
            **factory_kwargs
        )

        # Out proj from inner Mamba to d_inner
        self.dt_out_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        nn.init.zeros_(self.dt_out_proj.weight)  # start from bias-only for stable dt

        # Initialize dt_out_proj.bias like original Mamba dt bias
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
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
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


# ===================== Trainer =====================

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class PreNormResidual(nn.Module):
    """LayerNorm -> Block -> Dropout -> Residual"""
    def __init__(self, d_model, block: nn.Module, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.block = block
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.block(self.norm(x)))


class MambaClassifier(nn.Module):
    """Classifier for grayscale CIFAR-10 using Mamba or Mamba-in-Mamba"""
    def __init__(
        self,
        input_dim=1024,
        num_classes=10,
        d_model=128,
        n_layers=4,
        use_mamba_in_mamba=True,
        dropout=0.1,
        **mamba_kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.use_mamba_in_mamba = use_mamba_in_mamba

        self.input_proj = nn.Linear(1, d_model)  # grayscale pixel -> d_model
        trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)

        layers = []
        for i in range(n_layers):
            base_kwargs = dict(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
                dt_rank="auto",
                layer_idx=i,
            )
            if use_mamba_in_mamba:
                block = MambaInMamba(
                    **base_kwargs,
                    dt_mamba_d_state=mamba_kwargs.get("dt_mamba_d_state", 4),
                    dt_mamba_d_conv=mamba_kwargs.get("dt_mamba_d_conv", 2),
                    clamp_delta=mamba_kwargs.get("clamp_delta", 6.0),
                )
            else:
                block = Mamba(**base_kwargs)

            layers.append(PreNormResidual(d_model, block, dropout=dropout))

        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        b = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(b, -1, 1)  # (B, 32*32, 1)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

    def get_delta_maps(self):
        delta_maps = []
        for layer in self.layers:
            block = layer.block
            if hasattr(block, 'last_delta') and block.last_delta is not None:
                # reduce over d_inner
                delta = block.last_delta.mean(dim=1)  # (B, L)
                delta_maps.append(delta)
        return delta_maps


def visualize_delta(model, data_loader, device, save_path='delta_visualization.png'):
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            _ = model(images)
            delta_maps = model.get_delta_maps()

            if len(delta_maps) > 0:
                n_samples = min(4, images.size(0))
                n_layers = len(delta_maps)
                fig, axes = plt.subplots(n_samples, n_layers + 1, figsize=(3*(n_layers+1), 3*n_samples))

                for i in range(n_samples):
                    img = images[i].cpu().squeeze(0).numpy()
                    ax_img = axes[i, 0] if n_samples > 1 else axes[0]
                    ax_img.imshow(img, cmap='gray')
                    ax_img.set_title(f'Input (Label: {labels[i].item()})')
                    ax_img.axis('off')

                    for j, delta_map in enumerate(delta_maps):
                        delta_img = delta_map[i].cpu().numpy().reshape(32, 32)
                        denom = (delta_img.max() - delta_img.min())
                        delta_img = (delta_img - delta_img.min()) / (denom + 1e-8)
                        ax = axes[i, j+1] if n_samples > 1 else axes[j+1]
                        im = ax.imshow(delta_img, cmap='hot', vmin=0, vmax=1)
                        ax.set_title(f'Layer {j+1} Delta')
                        ax.axis('off')
                        if i == 0:
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Delta visualization saved to {save_path}")
                break
    model.train()


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, max_grad_norm=0.5):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)

        if torch.isnan(output).any():
            raise RuntimeError("NaN detected in model output")

        loss = criterion(output, target)

        if torch.isnan(loss):
            raise RuntimeError("NaN detected in loss")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    return total_loss / len(train_loader), correct / total


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return test_loss / len(test_loader), correct / total


def main():
    config = {
        'batch_size': 64,
        'epochs': 20,
        'lr': 3e-4,                    # slightly lower for stability
        'd_model': 128,
        'n_layers': 4,
        'dropout': 0.1,
        'use_mamba_in_mamba': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'save_dir': './checkpoints/cifar10_gray_mim',
    }

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    os.makedirs(config['save_dir'], exist_ok=True)

    # Data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    model_name = "MambaInMamba" if config['use_mamba_in_mamba'] else "Mamba"
    print(f"\nTraining {model_name} on CIFAR-10 (Grayscale)")
    print(f"Config: {config}\n")

    model = MambaClassifier(
        input_dim=1024,
        num_classes=10,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        use_mamba_in_mamba=config['use_mamba_in_mamba'],
        dt_mamba_d_state=4,
        dt_mamba_d_conv=2,
        clamp_delta=6.0,
    ).to(config['device'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    criterion = nn.CrossEntropyLoss()

    # AdamW with no weight decay on norms and biases
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if n.endswith("bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
    optimizer = optim.AdamW(
        [{'params': decay, 'weight_decay': 0.01},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=config['lr']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    best_acc = 0.0
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config['device'], epoch, max_grad_norm=0.5)
        test_loss, test_acc = test(model, test_loader, criterion, config['device'])
        scheduler.step()

        print(f'\nEpoch {epoch}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
        print(f'  Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc*100:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'config': config,
            }, os.path.join(config['save_dir'], f'{model_name}_best.pth'))
            print(f'  Saved best model with accuracy: {test_acc*100:.2f}%')

        if config['use_mamba_in_mamba'] and epoch % 3 == 0:
            visualize_delta(
                model,
                test_loader,
                config['device'],
                save_path=os.path.join(config['save_dir'], f'delta_epoch_{epoch}.png')
            )

    print(f'\nTraining completed! Best test accuracy: {best_acc*100:.2f}%')

    if config['use_mamba_in_mamba']:
        print("\nGenerating final delta visualization...")
        visualize_delta(
            model,
            test_loader,
            config['device'],
            save_path=os.path.join(config['save_dir'], 'delta_final.png')
        )


if __name__ == '__main__':
    main()
