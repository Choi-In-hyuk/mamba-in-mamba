# Comparing Mamba and Mamba-in-mamba(ours) in Mnist  

## Mamba  
<p float="left">
  <img src="https://github.com/user-attachments/assets/1bf6c860-deb7-4208-9526-37fed0decc27" width="250" />
  <img src="https://github.com/user-attachments/assets/1a372ea3-f318-49cf-b60e-00cc7ca2a199" width="250" />
  <img src="https://github.com/user-attachments/assets/811948f6-e0c1-4619-90e5-f9b41b2aa789" width="250" />
</p>

## Mamba-in-mamba (ours)  
<p float="left">
  <img src="https://github.com/user-attachments/assets/d1bdf7ef-cca3-4e40-a645-e509c8eceb26" width="250" />
  <img src="https://github.com/user-attachments/assets/74bc5ea2-6110-43d0-a161-ba02b07e16bd" width="250" />
  <img src="https://github.com/user-attachments/assets/e7b8707a-f2d2-4363-98c4-128f1b43b1e9" width="250" />
</p>

<p float="left">
  <img src="https://github.com/user-attachments/assets/b7ced2c0-dc71-4ce8-b9eb-b29351e4f3b5" width="760" />
</p>


# Comparing Mamba and Mamba-in-mamba(ours) in Cifar10  

## Mamba ： Best test accuracy: 63.04%  
<p float="left">
  <img src="https://github.com/user-attachments/assets/cd8c50c1-4abd-4865-b1a5-2b1756260a52" width="250" />
  <img src="https://github.com/user-attachments/assets/eead6285-1e72-4924-addc-e7acd4ee074f" width="250" />
  <img src="https://github.com/user-attachments/assets/c4de8bb7-857b-412a-8565-5018377a33c9" width="250" />
</p>

## Mamba-in-mamba (ours) : Best test accuracy: maybe 53?
<p float="left">
  <img src="https://github.com/user-attachments/assets/6dcc0044-d5dd-4475-9482-512526d0540c" width="250" />
  <img src="https://github.com/user-attachments/assets/50a73a54-a2ff-4f5b-9108-38c7ecb5c98c" width="250" />
  <img src="https://github.com/user-attachments/assets/446eb396-eaf7-44dc-81b2-d149337c426a" width="250" />
</p>

# Repository Structure (modified)

## Train: `~/mamba`
- **train_mnist.py**  
  Training script for MNIST using the **Mamba-in-Mamba** model  
- **train_mamba_mnist.py**  
  Training script for MNIST using the original **Mamba** model  

- **train_cifar10_gray_mamba_in_mamba.py**  
  Training script for CIFAR-10 (grayscale) using the **Mamba-in-Mamba** model  
- **train_cifar10_gray_mamba.py**  
  Training script for CIFAR-10 (grayscale) using the original **Mamba** model  


## Modules: `~/mamba/mamba_ssm/modules`
- **mamba_in_mamba.py**  
  Mamba-in-Mamba model implementation for MNIST experiments  
- **mamba_in_mamba_residual.py**  
  Residual-based Mamba-in-Mamba model implementation for CIFAR-10 experiments  

# Mamba-in-Mamba: Stability Improvements

## Overview
This document compares two versions of the Mamba-in-Mamba implementation, where the second version addresses NaN issues encountered during CIFAR-10 training.

## Key Changes Made

### 1. Weight Initialization Improvements

**Before (First Version):**
```python
# Used default PyTorch initialization
self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
```

**After (Second Version):**
```python
def trunc_normal_(w, std=0.02):
    nn.init.trunc_normal_(w, std=std)

# Small standard deviation initialization for all linear layers
self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
trunc_normal_(self.in_proj.weight, std=init_std)  # init_std=0.02
if self.in_proj.bias is not None:
    nn.init.zeros_(self.in_proj.bias)

# Initialize dt_out_proj weights to zero (bias-only start for stable dt)
nn.init.zeros_(self.dt_out_proj.weight)
```

### 2. Delta Value Clamping

**Before:**
```python
# No constraints on delta values
delta = self.dt_out_proj(dt_mamba_out)
delta = rearrange(delta, "(b l) d -> b d l", b=batch, l=seqlen)
```

**After:**
```python
# Clamp delta values to prevent extreme values
delta = self.dt_out_proj(dt_mamba_out)
delta = torch.clamp(delta, -self.clamp_delta, self.clamp_delta)  # clamp_delta=6.0
delta = rearrange(delta, "(b l) d -> b d l", b=Bsz, l=L)
```

### 3. Additional Stability Parameters

**New parameters added in the second version:**
```python
clamp_delta: float = 6.0,   # clamp pre-softplus delta to [-clamp_delta, clamp_delta]
init_std: float = 0.02,     # small init for projections
```

## Root Causes of NaN Issues

### 1. Initialization Problems
- **Issue**: Large initial weights → large delta values → extremely large values after softplus  
- **Solution**: Small standard deviation (0.02) initialization + zero initialization for dt_out_proj weights  

### 2. Delta Value Explosion
- **Issue**: Inner Mamba outputting extreme values → inf values after softplus  
- **Solution**: Apply `torch.clamp(-6.0, 6.0)` before softplus  

### 3. Gradient Instability
- **Issue**: Large delta values causing cascading gradient explosions  
- **Solution**: Conservative initialization + value constraints  

## Summary

The second version addresses NaN issues through **two simple but effective changes**:
1. **Better initialization**: Smaller initial weights and zero-initialized dt projection  
2. **Value clamping**: Preventing extreme delta values before softplus activation  

These minimal changes maintain the core Mamba-in-Mamba architecture while ensuring stable training on datasets like CIFAR-10.

