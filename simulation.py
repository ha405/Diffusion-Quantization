import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Import your modules
from networks import TDQ_MLP, SNR_TDQ_MLP
from scheduler import NoiseScheduler

# --- 1. The "Teacher" (Ground Truth Generator) ---
def find_optimal_interval(activation_tensor, num_candidates=200):
    """
    Finds the scalar 's' that minimizes L2 reconstruction error.
    """
    # Search range: from very small to the absolute max of the tensor
    abs_max = activation_tensor.abs().max().item()
    if abs_max == 0: return torch.tensor(1.0).to(activation_tensor.device)
    
    # We search ranges [0.1 * max ... 1.2 * max]
    candidates = torch.linspace(abs_max * 0.1, abs_max * 1.2, num_candidates).to(activation_tensor.device)
    
    best_loss = float('inf')
    best_s = candidates[-1]
    
    # Simulation of 4-bit Quantization (signed: -8 to 7)
    q_min, q_max = -8, 7
    
    for s in candidates:
        # Quantize
        x_scaled = activation_tensor / s
        x_int = x_scaled.round().clamp(q_min, q_max)
        x_recon = x_int * s
        
        # MSE Loss
        loss = torch.mean((activation_tensor - x_recon) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_s = s
            
    return best_s

# --- 2. Synthetic Data Generator ---
def generate_mock_diffusion_data(scheduler, num_samples=1000):
    """
    Generates synthetic activations that behave like a Diffusion Model.
    Variance roughly follows sqrt(1 - alpha_bar).
    """
    activations = []
    timesteps = []
    
    print(f"Generating synthetic data using {scheduler.schedule_type} schedule...")
    
    for t in range(0, scheduler.num_timesteps, scheduler.num_timesteps // num_samples):
        # 1. Get the variance scaling factor for this timestep
        # In real diffusion, layers usually scale with the noise level
        alpha_bar = scheduler.alphas_cumprod[t]
        
        # Physics approximation: Activation std dev is mixture of signal (1.0) and noise (sigma)
        # We simulate a layer where noise dominates at high t
        noise_std = torch.sqrt(1 - alpha_bar)
        signal_std = torch.sqrt(alpha_bar) * 0.5 # Arbitrary feature strength
        
        total_std = noise_std + signal_std
        
        # Create a random tensor with this std dev
        # Shape [128, 64] simulates a feature map
        data = torch.randn(128, 64) * total_std
        
        activations.append(data)
        timesteps.append(t)
        
    return activations, timesteps

# --- 3. Main Experiment Routine ---
def run_comparison():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # A. CONFIGURATION
    # We train on Linear, but we want to see if SNR works better
    train_scheduler = NoiseScheduler(schedule="linear")
    train_scheduler.schedule_type = "linear" # Just for logging

    # B. DATA GENERATION (The "Calibration" Phase)
    raw_acts, raw_ts = generate_mock_diffusion_data(train_scheduler, num_samples=200)
    
    print("Calculating Ground Truth Intervals (Teacher)...")
    gt_intervals = []
    for act in raw_acts:
        s = find_optimal_interval(act.to(device))
        gt_intervals.append(s)
    
    gt_intervals = torch.stack(gt_intervals).view(-1, 1) # [N, 1]
    train_ts_tensor = torch.tensor(raw_ts).to(device)    # [N]
    
    # C. MODEL SETUP
    model_baseline = TDQ_MLP().to(device)  # Uses Integer t
    model_snr = SNR_TDQ_MLP().to(device)   # Uses Log SNR
    
    opt_base = optim.Adam(model_baseline.parameters(), lr=0.005)
    opt_snr = optim.Adam(model_snr.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    
    # D. TRAINING LOOP
    print("\n--- Starting Training ---")
    epochs = 200
    
    # Pre-calculate inputs for SNR model
    # Note: We detach because we aren't training the scheduler, just the MLP
    train_log_snrs = train_scheduler.get_log_snr(train_ts_tensor.cpu()).to(device).view(-1, 1)

    for epoch in range(epochs):
        # 1. Train Baseline
        opt_base.zero_grad()
        pred_base = model_baseline(train_ts_tensor) # Input: Int T
        loss_base = loss_fn(pred_base, gt_intervals)
        loss_base.backward()
        opt_base.step()
        
        # 2. Train SNR Model
        opt_snr.zero_grad()
        pred_snr = model_snr(train_log_snrs)        # Input: Log SNR
        loss_snr = loss_fn(pred_snr, gt_intervals)
        loss_snr.backward()
        opt_snr.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Base Loss: {loss_base.item():.5f} | SNR Loss: {loss_snr.item():.5f}")

    # E. EVALUATION & VISUALIZATION
    print("\n--- Evaluation ---")
    
    model_baseline.eval()
    model_snr.eval()
    
    with torch.no_grad():
        # Plotting range 0 to 1000
        test_ts = torch.arange(0, 1000).to(device)
        
        # Baseline Predictions
        res_base = model_baseline(test_ts).cpu().numpy()
        
        # SNR Predictions
        test_log_snr = train_scheduler.get_log_snr(test_ts.cpu()).to(device).view(-1, 1)
        res_snr = model_snr(test_log_snr).cpu().numpy()
        
        # Ground Truth Points (for plotting)
        gt_y = gt_intervals.cpu().numpy()
        gt_x = train_ts_tensor.cpu().numpy()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(gt_x, gt_y, color='black', s=10, alpha=0.5, label='Ground Truth (from Activations)')
    plt.plot(res_base, color='red', linewidth=2, label='Baseline (Input: t)')
    plt.plot(res_snr, color='blue', linewidth=2, linestyle='--', label='Ours (Input: Log SNR)')
    
    plt.title(f"Quantization Interval Learning (Schedule: {train_scheduler.schedule_type})")
    plt.xlabel("Timestep (t)")
    plt.ylabel("Learned Interval (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("tdq_comparison.png")
    print("Saved plot to tdq_comparison.png")

    # F. ROBUSTNESS TEST (The "Killer" Feature)
    # What happens if we switch to Cosine schedule without retraining?
    print("\n--- Robustness Test (Cosine Schedule) ---")
    
    # Create a cosine scheduler
    cosine_scheduler = NoiseScheduler(schedule="cosine")
    
    # Get a specific step (e.g., t=500)
    t_test = torch.tensor([500]).to(device)
    
    # Baseline Output (Dependent only on t=500)
    base_out = model_baseline(t_test).item()
    
    # SNR Output (Dependent on the SNR at t=500 for Cosine)
    snr_val = cosine_scheduler.get_log_snr(t_test.cpu()).to(device).view(-1, 1)
    snr_out = model_snr(snr_val).item()
    
    print(f"At t=500:")
    print(f"Linear Noise Variance (approx): {1 - train_scheduler.alphas_cumprod[500]:.4f}")
    print(f"Cosine Noise Variance (approx): {1 - cosine_scheduler.alphas_cumprod[500]:.4f}")
    print("-" * 30)
    print(f"Baseline predicted s: {base_out:.4f} (Likely Wrong for Cosine)")
    print(f"SNR model predicted s: {snr_out:.4f} (Adaptive to noise level)")

if __name__ == "__main__":
    run_comparison()