# import torch
# import torch.nn as nn
# from diffusers import DDPMPipeline, DDPMScheduler
# import numpy as np
# import matplotlib.pyplot as plt

# from networks import TDQ_MLP, SNR_TDQ_MLP
# from scheduler import NoiseScheduler 

# MODEL_ID = "google/ddpm-cifar10-32" 
# TARGET_LAYER = "mid_block.resnets.0.conv1" 
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# class ActivationCollector:
#     def __init__(self):
#         self.activations = []
#         self.timesteps = []
#     def hook_fn(self, module, input, output):
#         self.activations.append(output.detach().cpu())

# def get_optimal_interval(activation, n_bits=4):
#     abs_max = activation.abs().max()
#     if abs_max == 0: return torch.tensor(1.0)
    
#     candidates = torch.linspace(abs_max * 0.1, abs_max * 1.1, 50)
#     best_loss = float('inf')
#     best_s = candidates[-1]
    
#     q_max = 2**(n_bits-1) - 1
#     q_min = -2**(n_bits-1)
    
#     for s in candidates:
#         x_q = (activation / s).round().clamp(q_min, q_max)
#         x_recon = x_q * s
#         loss = torch.mean((activation - x_recon)**2)
#         if loss < best_loss:
#             best_loss = loss
#             best_s = s    
#     return best_s

# def collect_real_data(scheduler_type="linear", num_images=4):
#     print(f"--- Collecting Data for Scheduler: {scheduler_type} ---")
#     pipeline = DDPMPipeline.from_pretrained(MODEL_ID).to(DEVICE)
#     pipeline.scheduler = DDPMScheduler(
#         beta_start=pipeline.scheduler.config.beta_start,
#         beta_end=pipeline.scheduler.config.beta_end,
#         beta_schedule=scheduler_type,
#         num_train_timesteps=pipeline.scheduler.config.num_train_timesteps)
#     collector = ActivationCollector()
#     model_scheduler_config = pipeline.scheduler.config
#     num_timesteps = model_scheduler_config.num_train_timesteps
#     beta_start = getattr(model_scheduler_config, "beta_start", 0.00085)
#     beta_end = getattr(model_scheduler_config, "beta_end", 0.0120)
#     # print(beta_start)
#     # print(beta_end)
#     target_module = pipeline.unet.mid_block.resnets[0].conv1
#     hook_handle = target_module.register_forward_hook(collector.hook_fn)
#     pipeline(batch_size=num_images, num_inference_steps=1000)
    
#     hook_handle.remove()
#     dataset = []    
#     total_steps = len(collector.activations)
#     print(f"captured {total_steps} steps of activations.")
    
#     results_t = []
#     results_s = []
    
#     for i, act in enumerate(collector.activations):
#         t = pipeline.scheduler.timesteps[i % len(pipeline.scheduler.timesteps)].item()
#         opt_s = get_optimal_interval(act)
#         results_t.append(t)
#         results_s.append(opt_s.item())
#         if i % 100 == 0:
#             print(f"Processed step {i}/{total_steps}")
#     return np.array(results_t), np.array(results_s), pipeline.scheduler, num_timesteps,beta_start,beta_end

# import torch
# import torch.nn as nn
from diffusers import StableDiffusionPipeline, DDPMScheduler
# import numpy as np
# import matplotlib.pyplot as plt

# from networks import TDQ_MLP, SNR_TDQ_MLP
# from scheduler import NoiseScheduler 

# MODEL_ID = "runwayml/stable-diffusion-v1-5" 
# TARGET_LAYER = "mid_block.resnets.0.conv1" 
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# class ActivationCollector:
#     def __init__(self):
#         self.activations = []
#         self.timesteps = []
#     def hook_fn(self, module, input, output):
#         self.activations.append(output.detach().cpu())

# def get_optimal_interval(activation, n_bits=4):
#     abs_max = activation.abs().max()
#     if abs_max == 0: return torch.tensor(1.0)
#     candidates = torch.linspace(abs_max * 0.1, abs_max * 1.1, 50)
#     best_loss = float('inf')
#     best_s = candidates[-1]
#     q_max = 2**(n_bits-1) - 1
#     q_min = -2**(n_bits-1)
#     for s in candidates:
#         x_q = (activation / s).round().clamp(q_min, q_max)
#         x_recon = x_q * s
#         loss = torch.mean((activation - x_recon)**2)
#         if loss < best_loss:
#             best_loss = loss
#             best_s = s    
#     return best_s

# def collect_real_data(scheduler_type="linear", num_images=4):
#     pipeline = StableDiffusionPipeline.from_pretrained(
#         MODEL_ID,
#         dtype=(torch.float16 if DEVICE.startswith("cuda") and torch.cuda.is_available() else torch.float32),
#         safety_checker=None,
#         requires_safety_checker=False
#     ).to(DEVICE)

#     # try:
#     pipeline.scheduler = DDPMScheduler(
#     beta_start=pipeline.scheduler.config.beta_start,
#     beta_end=pipeline.scheduler.config.beta_end,
#     beta_schedule=scheduler_type,
#     num_train_timesteps=pipeline.scheduler.config.num_train_timesteps
#     )
#     #     else:
#     #         pipeline.scheduler = pipeline.scheduler
#     # except Exception:
#     #     pipeline.scheduler = pipeline.scheduler
#     n_steps = 50
#     pipeline.scheduler.set_timesteps(n_steps)
#     model_scheduler_config = pipeline.scheduler.config
#     num_timesteps = model_scheduler_config.num_train_timesteps
#     beta_start = getattr(model_scheduler_config, "beta_start", 0.00085)
#     beta_end = getattr(model_scheduler_config, "beta_end", 0.0120)
#     print(pipeline.scheduler.config)
#     print(beta_start)
#     print(beta_end)
#     collector = ActivationCollector()
#     target_module = pipeline.unet.mid_block.resnets[0].conv1
#     hook_handle = target_module.register_forward_hook(collector.hook_fn)

#     prompts = [""] * num_images
#     generator = torch.Generator(device=DEVICE).manual_seed(0) if DEVICE.startswith("cuda") else torch.Generator().manual_seed(0)

#     pipeline(prompt=prompts, num_inference_steps=n_steps, guidance_scale=1.0, generator=generator)

#     hook_handle.remove()
#     total_steps = len(collector.activations)
#     results_t = []
#     results_s = []
#     timesteps = pipeline.scheduler.timesteps
#     for i, act in enumerate(collector.activations):
#         t = timesteps[i % len(timesteps)].item()
#         opt_s = get_optimal_interval(act)
#         results_t.append(t)
#         results_s.append(opt_s.item())
#         if i % 100 == 0:
#             print(f"Processed step {i}/{total_steps}")
#     return np.array(results_t), np.array(results_s), pipeline.scheduler,num_timesteps,beta_start,beta_end

import torch
import torch.nn as nn
from diffusers import DiTPipeline, DDIMScheduler
import numpy as np
import matplotlib.pyplot as plt

from networks import TDQ_MLP, SNR_TDQ_MLP
from scheduler import NoiseScheduler 

MODEL_ID = "facebook/DiT-XL-2-256"  
TARGET_LAYER = "transformer_blocks.0.attn1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ActivationCollector:
    def __init__(self):
        self.activations = []
        self.timesteps = []
    def hook_fn(self, module, input, output):
        self.activations.append(output.detach().cpu())

def get_optimal_interval(activation, n_bits=4):
    abs_max = activation.abs().max()
    if abs_max == 0: return torch.tensor(1.0)
    candidates = torch.linspace(abs_max * 0.1, abs_max * 1.1, 50)
    best_loss = float('inf')
    best_s = candidates[-1]
    q_max = 2**(n_bits-1) - 1
    q_min = -2**(n_bits-1)
    for s in candidates:
        x_q = (activation / s).round().clamp(q_min, q_max)
        x_recon = x_q * s
        loss = torch.mean((activation - x_recon)**2)
        if loss < best_loss:
            best_loss = loss
            best_s = s    
    return best_s

def _get_module_by_dotted_path(root, dotted):
    parts = dotted.split('.')
    cur = root
    for p in parts:
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    return cur

def collect_real_data(scheduler_type="linear", num_images=4):
    pipeline = DiTPipeline.from_pretrained(MODEL_ID, torch_dtype=(torch.float16 if DEVICE.startswith("cuda") and torch.cuda.is_available() else torch.float32)).to(DEVICE)
    pipeline.scheduler = DDPMScheduler(
        beta_start=pipeline.scheduler.config.beta_start,
        beta_end=pipeline.scheduler.config.beta_end,
        beta_schedule=scheduler_type,
        num_train_timesteps=pipeline.scheduler.config.num_train_timesteps)
    # for i, (name, module) in enumerate(pipeline.transformer.named_modules()):
    #     print(i, name, module.__class__.__name__)
    #     if i > 200:
    #         break
    n_steps = 50
    pipeline.scheduler.set_timesteps(n_steps)
    collector = ActivationCollector()
    model_scheduler_config = pipeline.scheduler.config
    num_timesteps = model_scheduler_config.num_train_timesteps
    beta_start = getattr(model_scheduler_config, "beta_start", 0.00085)
    beta_end = getattr(model_scheduler_config, "beta_end", 0.0120)
    collector = ActivationCollector()
    target_module = _get_module_by_dotted_path(pipeline.transformer, TARGET_LAYER)
    hook_handle = target_module.register_forward_hook(collector.hook_fn)
    words = ["golden retriever"]
    try:
        class_ids = pipeline.get_label_ids(words)
    except Exception:
        class_ids = [207]
    generator = torch.Generator(device=DEVICE).manual_seed(0) if DEVICE.startswith("cuda") else torch.Generator().manual_seed(0)
    pipeline(class_labels=None, num_inference_steps=50, generator=generator)
    hook_handle.remove()
    total_steps = len(collector.activations)
    results_t = []
    results_s = []
    timesteps = pipeline.scheduler.timesteps
    for i, act in enumerate(collector.activations):
        t = timesteps[i % len(timesteps)].item()
        opt_s = get_optimal_interval(act)
        results_t.append(t)
        results_s.append(opt_s.item())
        if i % 100 == 0:
            print(f"Processed step {i}/{total_steps}")
    return np.array(results_t), np.array(results_s), pipeline.scheduler, num_timesteps, beta_start, beta_end


