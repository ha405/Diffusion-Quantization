
import torch

class Config:
    MODEL_ID = "facebook/DiT-XL-2-256"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 
    IMAGE_SIZE = 256
    NUM_INFERENCE_STEPS = 50
    GUIDANCE_SCALE = 4.0
    TIMESTEP_INPUT_DIM = 1  
    TIMESTEP_EMBEDDING_DIM = 1152
    PREDICTOR_HIDDEN_DIM = 64
    PREDICTOR_OUTPUT_DIM = 2 
    OUTPUT_DIR = "./output_images"
