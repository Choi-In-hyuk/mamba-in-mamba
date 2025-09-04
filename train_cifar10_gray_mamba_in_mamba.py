# All code and comments in English only.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba_in_mamba_residual import MambaInMamba


def trunc_normal_(w, std=0.02):
    nn.init.trunc_normal_(w, std=std)


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
        'lr': 3e-4,
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
