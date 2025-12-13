import torch
import torch.nn as nn
import torch.optim as optim
import gc
import os
import json
import itertools
from diffusers import StableDiffusionPipeline, DDIMScheduler
from networks import SNR_TDQ_MLP, TDQ_MLP
from scheduler import NoiseScheduler

# Configuration
MODEL_ID = "runwayml/stable-diffusion-v1-5" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "quant_checkpoint_sd.pt"
DEBUG_JSON_PATH = "debug_stats.json"
BASELINE_DIR = "baseline_results"

NUM_CALIB_IMAGES = 64
NUM_INFERENCE_STEPS = 50
QUANT_METHOD = "snr"

CALIB_PROMPTS = [
    "A portrait of a cyberpunk robot in a neon city, 8k, highly detailed",
    "An oil painting of a cottage in the mountains with a river",
    "A cute golden retriever puppy running in a park, photorealistic",
    "Abstract geometric shapes floating in a void, matte painting",
    "A futuristic space station orbiting a blue planet",
    "A bowl of fresh fruit on a wooden table, still life",
    "A medieval knight in armor standing in a stone hallway",
    "A macro photograph of a water droplet on a green leaf",
    "An anime style illustration of a girl looking at the sunset",
    "A dark fantasy forest with glowing mushrooms and fog"
]

BASELINE_PROMPTS = [
    "A cyberpunk city street at night, neon lights, rain, highly detailed",
    "A cute golden retriever puppy running in a park, 4k",
    "An oil painting of a cottage in the mountains",
    "A futuristic robot portrait, matte painting"
]

def get_all_linear_layers(module, parent_name=""):
    layers = []
    for name, child in module.named_children():
        next_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, nn.Linear):
            layers.append(next_name)
        else:
            layers.extend(get_all_linear_layers(child, next_name))
    return layers

def get_module_by_name(root, dotted_path):
    cur = root
    for p in dotted_path.split('.'):
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    return cur

def get_optimal_interval(activation, n_bits=8):
    abs_max = activation.abs().max()
    if abs_max == 0:
        return torch.tensor(1.0, device=activation.device)
    
    q_max = 2**(n_bits-1) - 1
    q_min = -(2**(n_bits-1))
    
    base_scale = abs_max / q_max
    
    # Grid search for optimal scale to minimize MSE
    candidates = torch.linspace(base_scale * 0.3, base_scale * 1.2, 40, device=activation.device)
    
    best_loss = float('inf')
    best_s = candidates[-1]
    
    # Optimization: If tensor is huge, subsample for speed (optional, kept off for accuracy)
    check_act = activation

    for s in candidates:
        s = torch.max(s, torch.tensor(1e-9, device=s.device))
        x_q = (check_act / s).round().clamp(q_min, q_max)
        x_recon = x_q * s
        loss = torch.mean((check_act - x_recon)**2)
        if loss < best_loss:
            best_loss = loss
            best_s = s
            
    return best_s

class OnlineStatsCollector:
    def __init__(self, layer_names):
        # Stores just the float values to save memory
        self.stats = {ln: [] for ln in layer_names}

    def get_hook(self, layer_name):
        def hook(module, input, output):
            if not isinstance(output, torch.Tensor):
                return
            
            # Detach to ensure no gradients are tracked
            act = output.detach()
            s_val = get_optimal_interval(act, n_bits=8)
            # Store as python float
            self.stats[layer_name].append(s_val.cpu().item())
        return hook

def calibrate_and_save():
    print("Loading pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    
    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    pipe.scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",
        num_train_timesteps=1000
    )

    # --- 1. Generate Baseline Images ---
    print(f"Generating Baseline Images to {BASELINE_DIR}...")
    os.makedirs(BASELINE_DIR, exist_ok=True)
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    
    with torch.no_grad():
        images = pipe(prompt=BASELINE_PROMPTS, num_inference_steps=NUM_INFERENCE_STEPS, generator=generator).images
    
    for i, img in enumerate(images):
        save_path = os.path.join(BASELINE_DIR, f"baseline_{i}.png")
        img.save(save_path)
        print(f"Saved baseline: {save_path}")

    # --- 2. Prepare for Calibration ---
    all_linear_layers = get_all_linear_layers(pipe.unet)
    print(f"Found {len(all_linear_layers)} linear layers. Starting calibration...")

    collector = OnlineStatsCollector(all_linear_layers)
    handles = []
    
    for ln in all_linear_layers:
        try:
            mod = get_module_by_name(pipe.unet, ln)
            h = mod.register_forward_hook(collector.get_hook(ln))
            handles.append(h)
        except AttributeError:
            pass

    # --- 3. Run Calibration Inference (Batched) ---
    print(f"Running calibration inference on {NUM_CALIB_IMAGES} images...")
    
    # Iterator to cycle through real prompts
    prompt_iterator = itertools.cycle(CALIB_PROMPTS)
    
    # Batch size of 1 to prevent OOM
    batch_size = 1
    
    for i in range(0, NUM_CALIB_IMAGES, batch_size):
        # Get next prompts
        current_prompts = [next(prompt_iterator) for _ in range(batch_size)]
        
        with torch.no_grad():
            pipe(
                prompt=current_prompts, 
                num_inference_steps=NUM_INFERENCE_STEPS, 
                generator=generator
            )
        
        # Explicit garbage collection to keep memory clean
        if i % 10 == 0:
            print(f"Processed {i}/{NUM_CALIB_IMAGES} images...")
            gc.collect()
            if DEVICE == 'cuda': torch.cuda.empty_cache()

    # Remove hooks
    for h in handles: h.remove()

    # --- 4. Prepare Training Data ---
    timesteps_cpu = pipe.scheduler.timesteps.cpu()
    noise_sched = NoiseScheduler(num_timesteps=1000, schedule="linear")
    
    log_snr_unique = noise_sched.get_log_snr(timesteps_cpu)
    global_min_snr = log_snr_unique.min().item()
    global_max_snr = log_snr_unique.max().item()
    
    quant_registry = {
        "global_min_snr": global_min_snr,
        "global_max_snr": global_max_snr,
        "method": QUANT_METHOD,
        "layers": {}
    }
    
    debug_stats = {}

    print("Training MLPs...")
    layers_trained = 0
    
    # --- 5. Train MLPs per Layer ---
    for idx, layer_name in enumerate(all_linear_layers):
        s_targets_list = collector.stats[layer_name]
        
        if not s_targets_list:
            continue

        # Match collected stats to timesteps
        num_samples = len(s_targets_list)
        num_steps = len(timesteps_cpu)
        
        if num_samples == num_steps:
            t_vals = timesteps_cpu
        else:
            t_vals = []
            for i in range(num_samples):
                t_vals.append(timesteps_cpu[i % num_steps])
            t_vals = torch.tensor(t_vals)

        s_tensor = torch.tensor(s_targets_list, dtype=torch.float32).to(DEVICE).view(-1, 1)
        t_tensor = t_vals.to(DEVICE).long()
        
        # Choose Network Input (SNR or Time)
        if QUANT_METHOD == "snr":
            model_input = noise_sched.get_log_snr(t_tensor).view(-1, 1)
            mlp = SNR_TDQ_MLP().to(DEVICE)
        else:
            model_input = t_tensor.float().view(-1, 1)
            mlp = TDQ_MLP().to(DEVICE)

        opt = optim.Adam(mlp.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        
        final_loss = 0.0
        
        # Train MLP
        for _ in range(200):
            opt.zero_grad()
            pred = mlp(model_input)
            loss = loss_fn(pred, s_tensor)
            loss.backward()
            opt.step()
            final_loss = loss.item()

        # Save Weights
        quant_registry["layers"][layer_name] = mlp.state_dict()
        
        # Save Debug Stats
        debug_stats[layer_name] = {
            "min_scale": min(s_targets_list),
            "max_scale": max(s_targets_list),
            "final_mse_loss": final_loss,
            "samples_collected": len(s_targets_list)
        }

        layers_trained += 1
        
        if layers_trained % 50 == 0:
            print(f"Trained {layers_trained}/{len(all_linear_layers)} layers. Last Loss: {final_loss:.6f}")
            if DEVICE == 'cuda': torch.cuda.empty_cache()

    # --- 6. Save Results ---
    torch.save(quant_registry, SAVE_PATH)
    print(f"Calibration complete. Registry saved at {SAVE_PATH}")
    
    with open(DEBUG_JSON_PATH, "w") as f:
        json.dump(debug_stats, f, indent=4)
    print(f"Debug statistics saved at {DEBUG_JSON_PATH}")

if __name__ == "__main__":
    calibrate_and_save()