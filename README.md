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

