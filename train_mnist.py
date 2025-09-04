"""
MNIST training script for Mamba-in-Mamba architecture
Includes delta visualization to see where the model focuses attention
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
from einops import rearrange

# Import our Mamba-in-Mamba implementation
from mamba_ssm.modules.mamba_in_mamba import MambaInMamba
from mamba_ssm.modules.mamba_simple import Mamba  # For comparison


class MambaClassifier(nn.Module):
    """Simple classifier using Mamba blocks for MNIST"""
    
    def __init__(
        self, 
        input_dim=784,  # 28*28 for MNIST
        num_classes=10,
        d_model=128,
        n_layers=4,
        use_mamba_in_mamba=True,
        **mamba_kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.use_mamba_in_mamba = use_mamba_in_mamba
        
        # Input projection
        self.input_proj = nn.Linear(1, d_model)  # Project each pixel to d_model
        
        # Stack of Mamba layers
        self.layers = nn.ModuleList()
        self.layers = nn.ModuleList()
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
                # Pass inner-Mamba kwargs only to MambaInMamba
                layer = MambaInMamba(
                    **base_kwargs,
                    dt_mamba_d_state=mamba_kwargs.get("dt_mamba_d_state", 4),
                    dt_mamba_d_conv=mamba_kwargs.get("dt_mamba_d_conv", 2),
                )
            else:
                # Do NOT pass dt_mamba_* to plain Mamba
                layer = Mamba(**base_kwargs)
            self.layers.append(layer)
        
        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        """
        x: (batch, 784) for MNIST
        """
        batch_size = x.shape[0]
        
        # Reshape to sequence: (batch, seq_len, 1)
        x = x.view(batch_size, -1, 1)
        
        # Project to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Pass through Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classify
        x = self.norm(x)
        logits = self.classifier(x)
        
        return logits
    
    def get_delta_maps(self):
        """Extract delta values from all layers for visualization"""
        delta_maps = []
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'last_delta') and layer.last_delta is not None:
                # Average across d_inner dimension
                delta = layer.last_delta.mean(dim=1)  # (batch, seq_len)
                delta_maps.append(delta)
        return delta_maps


def visualize_delta(model, data_loader, device, save_path='delta_visualization.png'):
    """Visualize delta values to see where model focuses"""
    model.eval()
    
    # Get one batch
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(images.size(0), -1).to(device)
            
            # Forward pass
            _ = model(images)
            
            # Get delta maps
            delta_maps = model.get_delta_maps()
            
            if len(delta_maps) > 0:
                # Visualize first few samples
                n_samples = min(4, images.size(0))
                n_layers = len(delta_maps)
                
                fig, axes = plt.subplots(n_samples, n_layers + 1, figsize=(3*(n_layers+1), 3*n_samples))
                
                for i in range(n_samples):
                    # Show original image
                    img = images[i].cpu().numpy().reshape(28, 28)
                    if n_samples == 1:
                        ax_img = axes[0] if n_layers > 0 else axes
                    else:
                        ax_img = axes[i, 0]
                    ax_img.imshow(img, cmap='gray')
                    ax_img.set_title(f'Input (Label: {labels[i].item()})')
                    ax_img.axis('off')
                    
                    # Show delta maps for each layer
                    for j, delta_map in enumerate(delta_maps):
                        delta_img = delta_map[i].cpu().numpy().reshape(28, 28)
                        
                        # Normalize to [0, 1] for visualization
                        delta_img = (delta_img - delta_img.min()) / (delta_img.max() - delta_img.min() + 1e-8)
                        
                        if n_samples == 1:
                            ax = axes[j+1] if n_layers > 0 else axes
                        else:
                            ax = axes[i, j+1]
                        
                        # Use hot colormap: white (0) to red (1)
                        im = ax.imshow(delta_img, cmap='hot', vmin=0, vmax=1)
                        ax.set_title(f'Layer {j+1} Delta')
                        ax.axis('off')
                        
                        # Add colorbar for first sample
                        if i == 0:
                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.show()
                print(f"Delta visualization saved to {save_path}")
                break
    
    model.train()


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.view(data.size(0), -1).to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), correct / total


def test(model, test_loader, criterion, device):
    """Test the model"""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def main():
    # Configuration
    config = {
        'batch_size': 64,
        'epochs': 10,
        'lr': 1e-3,
        'd_model': 64,  # Smaller for MNIST
        'n_layers': 2,  # Fewer layers for simple task
        'use_mamba_in_mamba': False,  # Set to False to use regular Mamba
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'save_dir': './checkpoints/mamba',
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Model
    model_name = "MambaInMamba" if config['use_mamba_in_mamba'] else "Mamba"
    print(f"\nTraining {model_name} model on MNIST")
    print(f"Config: {config}\n")
    
    model = MambaClassifier(
        input_dim=784,
        num_classes=10,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        use_mamba_in_mamba=config['use_mamba_in_mamba'],
        dt_mamba_d_state=4,  # Small state for inner Mamba
        dt_mamba_d_conv=2,   # Small conv for inner Mamba
    ).to(config['device'])
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    best_acc = 0
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config['device'], epoch)
        test_loss, test_acc = test(model, test_loader, criterion, config['device'])
        scheduler.step()
        
        print(f'\nEpoch {epoch}/{config["epochs"]}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
        print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%')
        
        # Save best model
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
        
        # Visualize delta maps every few epochs
        if config['use_mamba_in_mamba'] and epoch % 3 == 0:
            visualize_delta(
                model, 
                test_loader, 
                config['device'], 
                save_path=os.path.join(config['save_dir'], f'delta_epoch_{epoch}.png')
            )
    
    print(f'\nTraining completed! Best test accuracy: {best_acc*100:.2f}%')
    
    # Final visualization
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