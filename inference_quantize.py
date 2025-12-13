import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from networks import SNR_TDQ_MLP, TDQ_MLP
from scheduler import NoiseScheduler
from quantize import QLayer, QuantGlobalContext

MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "quant_checkpoint_sd.pt"
OUTPUT_DIR = "quantized_results"
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGE_SIZE = 512
INFERENCE_SCHEDULE = "linear"

SKIP_PATTERNS = [
    "time_proj",
    "emb",
    "embedder",
    "norm",
]

# ---------------- UTILS ----------------
def set_nested_item(root, path, value):
    parts = path.split('.')
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    target_name = parts[-1]
    if target_name.isdigit():
        parent[int(target_name)] = value
    else:
        setattr(parent, target_name, value)

def get_module_by_name(root, dotted_path):
    parts = dotted_path.split('.')
    cur = root
    for p in parts:
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    return cur

# --- FIX 1: Pass context into patch_model ---
def patch_model(pipeline, checkpoint_data, global_context):
    print("Patching model with Quantization Wrappers...")
    method = checkpoint_data.get("method", "snr")
    layers_data = checkpoint_data["layers"]
    patched_count = 0
    
    for layer_name, mlp_state_dict in layers_data.items():
        if any(skip in layer_name for skip in SKIP_PATTERNS):
            continue

        try:
            original_module = get_module_by_name(pipeline.unet, layer_name)

            if not isinstance(original_module, nn.Linear):
                continue

            if method == "snr":
                mlp = SNR_TDQ_MLP().to(DEVICE)
            else:
                mlp = TDQ_MLP().to(DEVICE)
            
            mlp.load_state_dict(mlp_state_dict)

            # --- FIX 2: Pass context to QLayer ---
            # Assuming QLayer constructor accepts 'context' or has a set_context method
            # If your QLayer definition doesn't accept context in __init__, 
            # you might need: q_layer.context = global_context
            q_layer = QLayer(original_module, mlp, abits=8, wbits=8) 
            
            # Manually inject context if not in init
            if hasattr(q_layer, 'context'):
                q_layer.context = global_context 
            else:
                # Fallback: Many implementations pass it in __init__
                # You must ensure your QLayer class stores this context!
                pass 

            q_layer = q_layer.to(DEVICE)

            # --- FIX 3: Enable Activation Quantization ---
            q_layer.quantize_w = True
            q_layer.quantize_act = True  # <--- CRITICAL FIX: Must be True for TDQ

            set_nested_item(pipeline.unet, layer_name, q_layer)
            patched_count += 1
            
        except AttributeError as e:
            print(f"Failed to patch {layer_name}: {e}")
            pass
            
    print(f"Successfully patched {patched_count} layers.")
    return pipeline

@torch.no_grad()
def run_quantized_inference(prompts):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError("Run calibrate.py first!")
    
    print(f"Loading Base Model: {MODEL_ID}")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        torch_dtype=(torch.float16 if DEVICE=="cuda" else torch.float32)
    ).to(DEVICE)

    # --- FIX 4: Initialize Context BEFORE patching ---
    # This ensures the layers and the loop share the same context object
    context = QuantGlobalContext()
    
    ckpt = torch.load(CHECKPOINT_PATH)
    pipe = patch_model(pipe, ckpt, context)

    pipe.scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",
        num_train_timesteps=1000 
    )
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    noise_sched = NoiseScheduler(
        num_timesteps=1000, 
        schedule=INFERENCE_SCHEDULE
    )
    
    batch_size = len(prompts)

    text_inputs = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(DEVICE))[0]
    
    uncond_input = pipe.tokenizer([""] * batch_size, padding="max_length", max_length=text_inputs.input_ids.shape[-1], return_tensors="pt")
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(DEVICE))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, pipe.unet.config.in_channels, IMAGE_SIZE // 8, IMAGE_SIZE // 8),
        device=DEVICE,
        dtype=pipe.unet.dtype
    )
    
    print(f"Starting Generation ({INFERENCE_SCHEDULE})...")
    
    for i, t in enumerate(pipe.scheduler.timesteps):
        t_scalar = torch.tensor([t], device=DEVICE, dtype=torch.long)
        current_input = noise_sched.get_log_snr(t_scalar)
        
        # Update the context that is linked to the QLayers
        context.set_current_snr(current_input)
        
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings
        ).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
        
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Stats (Optional: wrap in try/catch in case QLayer doesn't push stats)
        try:
            avg_err, max_err = context.get_and_reset_error_stats()
            if i % 10 == 0:
                print(f"Step {i} | Quant MSE: {avg_err:.6e} (Max: {max_err:.6e})")
        except:
            pass
            
        if i % 10 == 0:
            print(f"Step {i}/{NUM_INFERENCE_STEPS} | SNR: {current_input.item():.4f}")

    print("Decoding images...")
    # --- FIX 5: Don't force float32 unless necessary ---
    # pipe.vae = pipe.vae.to(torch.float32) 
    
    # Calculate Latents
    latents = latents / pipe.vae.config.scaling_factor
    images = pipe.vae.decode(latents).sample
 
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy() # Ensure float before numpy
    images = (images * 255).round().astype("uint8")
    
    for idx, img_arr in enumerate(images):
        pil_img = Image.fromarray(img_arr)
        save_path = os.path.join(OUTPUT_DIR, f"sd_quant_{idx}.png")
        pil_img.save(save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    prompts = [
        "A cyberpunk city street at night, neon lights, rain, highly detailed",
        "A cute golden retriever puppy running in a park, 4k",
        "An oil painting of a cottage in the mountains",
        "A futuristic robot portrait, matte painting"
    ]
    run_quantized_inference(prompts)