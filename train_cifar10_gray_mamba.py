"""
CIFAR-10 (Grayscale) training script for Mamba architecture with delta visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from mamba_ssm.modules.mamba_simple import Mamba


class MambaClassifier(nn.Module):
    def __init__(self, input_dim=1024, num_classes=10, d_model=128, n_layers=4, **mamba_kwargs):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
                dt_rank="auto",
                layer_idx=i
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)  # (B, 32*32, 1)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        logits = self.classifier(x)
        return logits

    def get_delta_maps(self):
        delta_maps = []
        for layer in self.layers:
            if hasattr(layer, 'last_delta') and layer.last_delta is not None:
                delta = layer.last_delta.mean(dim=1)
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
                        delta_img = (delta_img - delta_img.min()) / (delta_img.max() - delta_img.min() + 1e-8)
                        ax = axes[i, j+1] if n_samples > 1 else axes[j+1]
                        im = ax.imshow(delta_img, cmap='hot', vmin=0, vmax=1)
                        ax.set_title(f'Layer {j+1} Delta')
                        ax.axis('off')
                        if i == 0:
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.show()
                print(f"Delta visualization saved to {save_path}")
                break
    model.train()


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(train_loader), correct / total


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy


def main():
    config = {
        'batch_size': 64,
        'epochs': 20,
        'lr': 1e-3,
        'd_model': 64,
        'n_layers': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'save_dir': './checkpoints/cifar10_gray_mamba',
    }

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    os.makedirs(config['save_dir'], exist_ok=True)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"\nTraining Mamba model on CIFAR-10 (Grayscale)")
    print(f"Config: {config}\n")

    model = MambaClassifier(
        input_dim=1024,
        num_classes=10,
        d_model=config['d_model'],
        n_layers=config['n_layers']
    ).to(config['device'])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    best_acc = 0
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config['device'], epoch)
        test_loss, test_acc = test(model, test_loader, criterion, config['device'])
        scheduler.step()

        print(f'\nEpoch {epoch}/{config["epochs"]}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
        print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'config': config,
            }, os.path.join(config['save_dir'], f'Mamba_best.pth'))
            print(f'  Saved best model with accuracy: {test_acc*100:.2f}%')

        if epoch % 3 == 0:
            visualize_delta(
                model,
                test_loader,
                config['device'],
                save_path=os.path.join(config['save_dir'], f'delta_epoch_{epoch}.png')
            )

    print(f'\nTraining completed! Best test accuracy: {best_acc*100:.2f}%')

    print("\nGenerating final delta visualization...")
    visualize_delta(
        model,
        test_loader,
        config['device'],
        save_path=os.path.join(config['save_dir'], 'delta_final.png')
    )


if __name__ == '__main__':
    main()
