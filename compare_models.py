"""
Compare Mamba vs Mamba-in-Mamba performance
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from mamba_ssm.modules.mamba_in_mamba import MambaInMamba
from mamba_ssm.modules.mamba_simple import Mamba
from train_mnist import MambaClassifier, test


def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, test_loader, device, num_runs=3):
    """Measure average inference time"""
    model.eval()
    times = []
    
    with torch.no_grad():
        # Warmup
        for data, _ in test_loader:
            data = data.view(data.size(0), -1).to(device)
            _ = model(data)
            break
        
        # Actual measurement
        for run in range(num_runs):
            start_time = time.time()
            for data, _ in test_loader:
                data = data.view(data.size(0), -1).to(device)
                _ = model(data)
            end_time = time.time()
            times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)


def compare_models():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    d_model = 64
    n_layers = 2
    
    print("=" * 60)
    print("Mamba vs Mamba-in-Mamba Comparison")
    print("=" * 60)
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize both models
    print("\n1. Model Architecture Comparison:")
    print("-" * 40)
    
    # Regular Mamba
    model_mamba = MambaClassifier(
        input_dim=784,
        num_classes=10,
        d_model=d_model,
        n_layers=n_layers,
        use_mamba_in_mamba=False
    ).to(device)
    
    # Mamba-in-Mamba
    model_mim = MambaClassifier(
        input_dim=784,
        num_classes=10,
        d_model=d_model,
        n_layers=n_layers,
        use_mamba_in_mamba=True,
        dt_mamba_d_state=4,
        dt_mamba_d_conv=2,
    ).to(device)
    
    # Parameter count
    params_mamba = count_parameters(model_mamba)
    params_mim = count_parameters(model_mim)
    
    print(f"Regular Mamba parameters: {params_mamba:,}")
    print(f"Mamba-in-Mamba parameters: {params_mim:,}")
    print(f"Parameter increase: {(params_mim/params_mamba - 1)*100:.1f}%")
    
    # Train both models from scratch for fair comparison
    print("\n2. Training Performance:")
    print("-" * 40)
    
    criterion = nn.CrossEntropyLoss()
    epochs = 5  # Quick training for comparison
    
    # Train regular Mamba
    print("\nTraining Regular Mamba...")
    optimizer = torch.optim.AdamW(model_mamba.parameters(), lr=1e-3)
    train_dataset = datasets.MNIST('./data', train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    mamba_train_times = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        model_mamba.train()
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', leave=False):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model_mamba(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_mamba.parameters(), 1.0)
            optimizer.step()
        mamba_train_times.append(time.time() - start)
    
    # Train Mamba-in-Mamba
    print("\nTraining Mamba-in-Mamba...")
    optimizer = torch.optim.AdamW(model_mim.parameters(), lr=1e-3)
    
    mim_train_times = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        model_mim.train()
        for data, target in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', leave=False):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model_mim(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_mim.parameters(), 1.0)
            optimizer.step()
        mim_train_times.append(time.time() - start)
    
    print(f"\nAvg training time per epoch:")
    print(f"Regular Mamba: {np.mean(mamba_train_times):.2f}s ± {np.std(mamba_train_times):.2f}s")
    print(f"Mamba-in-Mamba: {np.mean(mim_train_times):.2f}s ± {np.std(mim_train_times):.2f}s")
    print(f"Training overhead: {(np.mean(mim_train_times)/np.mean(mamba_train_times) - 1)*100:.1f}%")
    
    # Test accuracy
    print("\n3. Test Accuracy (after quick training):")
    print("-" * 40)
    
    _, acc_mamba = test(model_mamba, test_loader, criterion, device)
    _, acc_mim = test(model_mim, test_loader, criterion, device)
    
    print(f"Regular Mamba: {acc_mamba*100:.2f}%")
    print(f"Mamba-in-Mamba: {acc_mim*100:.2f}%")
    
    # Inference speed
    print("\n4. Inference Speed:")
    print("-" * 40)
    
    time_mamba, std_mamba = measure_inference_time(model_mamba, test_loader, device)
    time_mim, std_mim = measure_inference_time(model_mim, test_loader, device)
    
    print(f"Regular Mamba: {time_mamba:.3f}s ± {std_mamba:.3f}s")
    print(f"Mamba-in-Mamba: {time_mim:.3f}s ± {std_mim:.3f}s")
    print(f"Inference overhead: {(time_mim/time_mamba - 1)*100:.1f}%")
    
    # Load best checkpoint if available
    print("\n5. Best Model Performance (if checkpoint exists):")
    print("-" * 40)
    
    try:
        checkpoint = torch.load('./checkpoints/MambaInMamba_best.pth', map_location=device)
        model_mim_best = MambaClassifier(
            input_dim=784,
            num_classes=10,
            d_model=d_model,
            n_layers=n_layers,
            use_mamba_in_mamba=True,
            dt_mamba_d_state=4,
            dt_mamba_d_conv=2,
        ).to(device)
        model_mim_best.load_state_dict(checkpoint['model_state_dict'])
        
        _, acc_best = test(model_mim_best, test_loader, criterion, device)
        print(f"Mamba-in-Mamba (best checkpoint): {acc_best*100:.2f}%")
        print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    except:
        print("No checkpoint found")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Parameter overhead: +{(params_mim/params_mamba - 1)*100:.1f}%")
    print(f"Training time overhead: +{(np.mean(mim_train_times)/np.mean(mamba_train_times) - 1)*100:.1f}%")
    print(f"Inference time overhead: +{(time_mim/time_mamba - 1)*100:.1f}%")
    print(f"Accuracy difference: {(acc_mim - acc_mamba)*100:+.2f}%")
    
    # Visualization comparison
    print("\nGenerating comparison plot...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Parameters
    axes[0].bar(['Mamba', 'Mamba-in-Mamba'], [params_mamba, params_mim])
    axes[0].set_ylabel('Parameters')
    axes[0].set_title('Model Size')
    
    # Training time
    axes[1].bar(['Mamba', 'Mamba-in-Mamba'], 
                [np.mean(mamba_train_times), np.mean(mim_train_times)])
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Training Time per Epoch')
    
    # Accuracy
    axes[2].bar(['Mamba', 'Mamba-in-Mamba'], 
                [acc_mamba*100, acc_mim*100])
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Test Accuracy')
    axes[2].set_ylim([90, 100])
    
    plt.tight_layout()
    plt.savefig('./checkpoints/comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Comparison plot saved to ./checkpoints/comparison.png")


if __name__ == '__main__':
    compare_models()