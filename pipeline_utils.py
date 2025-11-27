
import torch
import os
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from config import Config

class DiTInfrastructure:
    def __init__(self):
        self.config = Config()
        self.pipe = None
        
    def load_pipeline(self):
        print(f"Loading DiT model: {self.config.MODEL_ID}...")
        
        self.pipe = DiTPipeline.from_pretrained(
            self.config.MODEL_ID, 
            torch_dtype=self.config.DTYPE
        )

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(self.config.DEVICE)
        print("Model loaded successfully.")

    def generate_baseline_images(self, class_labels, seed=42):
        if self.pipe is None:
            self.load_pipeline()
            
        generator = torch.manual_seed(seed)
        
        print(f"Generating baseline images for classes: {class_labels}")

        output = self.pipe(
            class_labels=class_labels,
            num_inference_steps=self.config.NUM_INFERENCE_STEPS,
            guidance_scale=self.config.GUIDANCE_SCALE,
            generator=generator
        )
        
        images = output.images
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        paths = []
        for i, (img, label) in enumerate(zip(images, class_labels)):
            path = f"{self.config.OUTPUT_DIR}/baseline_class_{label}_{i}.png"
            img.save(path)
            paths.append(path)
            
        print(f"Saved {len(images)} images to {self.config.OUTPUT_DIR}")
        return images, paths
