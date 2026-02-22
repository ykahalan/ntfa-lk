import torch
import numpy as np
import matplotlib.pyplot as plt
from ntfa_lk import NTFALayer

if __name__ == "__main__":
    print("=" * 70)
    print("NTFA-LK Example: Noisy Signal Decomposition and Recovery")
    print("(PyTorch version with cubic interpolation)")
    print("=" * 70)
    
    # Initialize
    step_size = 4
    alpha = 0.001
    beta = 0.03
    window_size = 20
    Fs = 100
    t = np.arange(0, 5, 1/Fs)
    
    # Generate signal
    print("Generating test signal (3 Hz + 7 Hz with Gaussian noise)...")
    signal = np.sin(2*np.pi*3*t) + 0.5*np.sin(2*np.pi*7*t)
    signal = signal + 0.2*np.random.randn(len(signal))
    
    # Convert to PyTorch
    signal_torch = torch.from_numpy(signal).float()
    
    # Create NTFA-LK layer with hybrid kernel
    print("Applying NTFA-LK with hybrid kernel...")
    layer = NTFALayer(
        kernel_names=["polynomial", "gaussian"],
        gamma=[0.5, 0.5],
        window_size=window_size,
        step_size=step_size,
        alpha=alpha,
        beta=beta,
        interp_factor=0.125
    )
    
    # Forward pass + inverse transform
    with torch.no_grad():
        tfr = layer(signal_torch)               # shape: (1, n_windows, n_freqs)
        recovered = layer.inverse_transform(tfr) # shape: (1, n_windows)
    
    # Drop batch dim for numpy/plotting
    tfr_np = tfr.squeeze(0).numpy()
    recovered_np = recovered.squeeze(0).numpy()

    print(f"Original signal shape:  {signal.shape}")
    print(f"TFR shape:              {tfr_np.shape}")
    print(f"Recovered signal shape: {recovered_np.shape}")
    
    # --- Plot 1: TFR heatmap ---
    print("Generating plots...")
    plt.figure(figsize=(10, 6))
    plt.imshow(tfr_np, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.title('NTFA-LK Time-Frequency Representation (with cubic interpolation)')
    plt.xlabel('Frequency bins')
    plt.ylabel('Time windows')
    plt.tight_layout()

    # --- Plot 2: Original, TFR sum, and recovered signal ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    axes[0].plot(signal, 'r-', alpha=0.6, linewidth=0.8, label='Original noisy signal')
    axes[0].set_title('Original Noisy Signal')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(tfr_np.sum(axis=1), 'b-', linewidth=1.5, label='NTFA-LK TFR (sum over freqs)')
    axes[1].set_title('NTFA-LK TFR — Frequency Energy per Window')
    axes[1].set_xlabel('Time windows')
    axes[1].set_ylabel('Magnitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(recovered_np, 'g-', linewidth=1.5, label='Recovered signal (inverse transform)')
    axes[2].set_title('Recovered Signal via Inverse Transform')
    axes[2].set_xlabel('Time windows')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    
    print("=" * 70)
    print("NTFA-LK processing complete")
    print("=" * 70)
