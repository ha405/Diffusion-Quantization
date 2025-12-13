import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from networks import TDQ_MLP, SNR_TDQ_MLP
from scheduler import NoiseScheduler 
from diffusers import DiTPipeline, DDPMScheduler

MODEL_ID = "facebook/DiT-XL-2-256"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROBE_LAYERS = [
    "transformer_blocks.0.attn1.to_q",
    "transformer_blocks.0.ff.net.0.proj",
    "transformer_blocks.7.attn1.to_q",
    "transformer_blocks.7.ff.net.0.proj",
    "transformer_blocks.14.attn1.to_q",
    "transformer_blocks.14.ff.net.0.proj",
    "transformer_blocks.21.attn1.to_q",
    "transformer_blocks.21.ff.net.0.proj",
    "transformer_blocks.27.attn1.to_q",
    "transformer_blocks.27.ff.net.0.proj",
]

class MultiActivationCollector:
    def __init__(self, layer_list):
        self.activations = {ln: [] for ln in layer_list}
    
    def hook_fn(self, layer_name):
        def _hook(module, input, output):
            self.activations[layer_name].append(output.detach().cpu())
        return _hook

def _get_module_by_dotted_path(root, dotted):
    parts = dotted.split('.')
    cur = root
    for p in parts:
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    return cur

def get_optimal_interval(activation, n_bits=4):
    abs_max = activation.abs().max()
    if abs_max == 0:
        return torch.tensor(1.0)
    candidates = torch.linspace(abs_max * 0.1, abs_max * 1.1, 50)
    best_loss = float('inf')
    best_s = candidates[-1]
    q_max = 2**(n_bits-1) - 1
    q_min = -2**(n_bits-1)
    
    if activation.numel() > 100000:
        check_act = activation.flatten()[::10]
    else:
        check_act = activation

    for s in candidates:
        x_q = (check_act / s).round().clamp(q_min, q_max)
        x_recon = x_q * s
        loss = torch.mean((check_act - x_recon)**2)
        if loss < best_loss:
            best_loss = loss
            best_s = s
    return best_s

def collect_real_data(layer_list, scheduler_type="linear", num_images=4):
    print(f"--- Collecting Data for Scheduler: {scheduler_type} ---")
    pipeline = DiTPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=(torch.float16 if DEVICE.startswith("cuda") and torch.cuda.is_available() else torch.float32)
    ).to(DEVICE)

    pipeline.scheduler = DDPMScheduler(
        beta_start=pipeline.scheduler.config.beta_start,
        beta_end=pipeline.scheduler.config.beta_end,
        beta_schedule=scheduler_type,
        num_train_timesteps=pipeline.scheduler.config.num_train_timesteps
    )

    n_steps = 50
    pipeline.scheduler.set_timesteps(n_steps)

    model_scheduler_config = pipeline.scheduler.config
    num_timesteps = model_scheduler_config.num_train_timesteps
    beta_start = getattr(model_scheduler_config, "beta_start", 0.00085)
    beta_end = getattr(model_scheduler_config, "beta_end", 0.0120)

    collector = MultiActivationCollector(layer_list)

    hook_handles = []
    registered_layers = []
    
    for ln in layer_list:
        try:
            module = _get_module_by_dotted_path(pipeline.transformer, ln)
            hook_handles.append(module.register_forward_hook(collector.hook_fn(ln)))
            registered_layers.append(ln)
        except AttributeError:
            print(f"Warning: Layer '{ln}' not found. Skipping.")
        except Exception:
            pass

    generator = torch.Generator(device=DEVICE).manual_seed(0)
    class_labels = torch.randint(0, 1000, (num_images,), device=DEVICE)
    
    with torch.no_grad():
        pipeline(class_labels=class_labels, num_inference_steps=n_steps, generator=generator)

    for h in hook_handles:
        h.remove()

    layer_results_t = {}
    layer_results_s = {}
    timesteps = pipeline.scheduler.timesteps.cpu().numpy()

    print("Computing optimal intervals per layer...")
    
    for ln in registered_layers:
        acts = collector.activations[ln]
        if not acts:
            continue
            
        t_list = []
        s_list = []
        
        for i, act in enumerate(acts):
            t = timesteps[i % len(timesteps)]
            opt_s = get_optimal_interval(act)
            t_list.append(t)
            s_list.append(opt_s.item())
            
        if len(t_list) > 0:
            layer_results_t[ln] = np.array(t_list)
            layer_results_s[ln] = np.array(s_list)
            print(f"Processed stats for {ln} ({len(t_list)} steps)")

    return layer_results_t, layer_results_s, pipeline.scheduler, num_timesteps, beta_start, beta_end

def run():
    print(f"Running on {DEVICE}")
    
    print("\n>>> STEP 1: Collecting Training Data (Linear Schedule)...")
    train_t_dict, train_s_dict, _, num_ts, bstart, bend = collect_real_data(
        PROBE_LAYERS, scheduler_type="linear", num_images=10
    )
    
    if not train_t_dict:
        print("ERROR: No training data collected. Exiting.")
        return

    print("\n>>> STEP 2: Collecting Test Data (Sigmoid Schedule)...")
    test_t_dict, test_s_dict, _, _, _, _ = collect_real_data(
        PROBE_LAYERS, scheduler_type="sigmoid", num_images=10
    )

    lin_scheduler = NoiseScheduler(num_timesteps=num_ts, beta_start=bstart, beta_end=bend, schedule="linear")
    sig_scheduler = NoiseScheduler(num_timesteps=num_ts, beta_start=bstart, beta_end=bend, schedule="sigmoid")

    any_layer = list(train_t_dict.keys())[0]
    
    lin_t_np = train_t_dict[any_layer]
    lin_t_tensor = torch.tensor(lin_t_np, dtype=torch.long).to(DEVICE)
    lin_log_snr = lin_scheduler.get_log_snr(lin_t_tensor.cpu()).to(DEVICE).view(-1, 1)

    train_min_snr = lin_log_snr.min().item()
    train_max_snr = lin_log_snr.max().item()
    print(f"\n[Info] Training Log-SNR Range: [{train_min_snr:.4f}, {train_max_snr:.4f}]")

    if any_layer not in test_t_dict:
        any_layer = list(test_t_dict.keys())[0]

    sig_t_np = test_t_dict[any_layer]
    sig_t_tensor = torch.tensor(sig_t_np, dtype=torch.long).to(DEVICE)
    sig_log_snr = sig_scheduler.get_log_snr(sig_t_tensor.cpu()).to(DEVICE).view(-1, 1)

    # NO CLAMPING HERE - PASSING RAW OOD INPUTS

    results_base_lin = []
    results_ours_lin = []
    results_base_sig = []
    results_ours_sig = []

    for layer_name in PROBE_LAYERS:
        if layer_name not in train_s_dict or layer_name not in test_s_dict:
            continue

        print(f"\n=== Processing Layer: {layer_name} ===")
        
        s_train_target = torch.tensor(train_s_dict[layer_name], dtype=torch.float32).to(DEVICE).view(-1, 1)
        s_test_target_np = test_s_dict[layer_name]

        baseline_mlp = TDQ_MLP().to(DEVICE)
        snr_mlp = SNR_TDQ_MLP().to(DEVICE)
        
        opt_base = optim.Adam(baseline_mlp.parameters(), lr=1e-3)
        opt_snr = optim.Adam(snr_mlp.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        
        for epoch in range(300):
            opt_base.zero_grad()
            out_base = baseline_mlp(lin_t_tensor)
            loss_base = loss_fn(out_base, s_train_target)
            loss_base.backward()
            opt_base.step()
            
            opt_snr.zero_grad()
            out_snr = snr_mlp(lin_log_snr)
            loss_snr = loss_fn(out_snr, s_train_target)
            loss_snr.backward()
            opt_snr.step()

        with torch.no_grad():
            pred_base_lin = baseline_mlp(lin_t_tensor).cpu().numpy().flatten()
            pred_snr_lin = snr_mlp(lin_log_snr).cpu().numpy().flatten()
            
            pred_base_sig = baseline_mlp(sig_t_tensor).cpu().numpy().flatten()
            
            # Use RAW Unclamped input
            pred_snr_sig = snr_mlp(sig_log_snr).cpu().numpy().flatten()

        mse_base_lin = np.mean((pred_base_lin - train_s_dict[layer_name])**2)
        mse_snr_lin = np.mean((pred_snr_lin - train_s_dict[layer_name])**2)
        
        mse_base_sig = np.mean((pred_base_sig - s_test_target_np)**2)
        mse_snr_sig = np.mean((pred_snr_sig - s_test_target_np)**2)

        print(f"  Linear MSE  -> Base: {mse_base_lin:.5f} | Ours: {mse_snr_lin:.5f}")
        print(f"  Sigmoid MSE -> Base: {mse_base_sig:.5f} | Ours: {mse_snr_sig:.5f}")

        results_base_lin.append(mse_base_lin)
        results_ours_lin.append(mse_snr_lin)
        results_base_sig.append(mse_base_sig)
        results_ours_sig.append(mse_snr_sig)

    print("\n" + "="*50)
    print("FINAL RESULTS (Mean across all layers) - NO CLAMPING")
    print("="*50)
    print(f"LINEAR (Train Schedule):")
    print(f"  Avg Baseline MSE: {np.mean(results_base_lin):.5f}")
    print(f"  Avg Ours MSE:     {np.mean(results_ours_lin):.5f}")
    
    print(f"\nSIGMOID (Test Schedule - OOD):")
    print(f"  Avg Baseline MSE: {np.mean(results_base_sig):.5f}")
    print(f"  Avg Ours MSE:     {np.mean(results_ours_sig):.5f}")
    print("="*50)

if __name__ == "__main__":
    run()