# mypy: ignore-errors

import argparse
import gc
import os
import sys

import torch
from accelerate import Accelerator
from huggingface_hub import HfFolder
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
    MllamaProcessor,
)

# Force PyTorch to be more aggressive with memory cleanup
torch.cuda.empty_cache()
gc.collect()

# Set environment variables for CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Initialize accelerator
accelerator = Accelerator(
    mixed_precision="bf16",
)
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = 1120


def get_hf_token():
    """Retrieve Hugging Face token from the cache or environment."""
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token
    token = HfFolder.get_token()
    if token:
        return token
    print("Hugging Face token not found. Please login using `huggingface-cli login`.")
    sys.exit(1)


def load_model_and_processor(model_name: str):
    """Load model and processor from Huggingface Hub"""
    print(f"Loading model: {model_name}")
    hf_token = get_hf_token()

    # Clear memory before loading
    torch.cuda.empty_cache()
    gc.collect()

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    try:
        # Load model with simpler device mapping
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            token=hf_token,
            low_cpu_mem_usage=True,
        )

        # Load processor
        processor = MllamaProcessor.from_pretrained(
            model_name,
            token=hf_token,
            use_safetensors=True,
        )

        model = model.eval()  # Set to evaluation mode

        # Move model to device if not already there
        if not hasattr(model, "device_map"):
            model = model.to(device)

        # Clear memory after loading
        torch.cuda.empty_cache()
        gc.collect()

        return model, processor

    except Exception as e:
        print(f"Error loading model: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        raise


def resize_image(img: Image.Image, max_dim: int = MAX_IMAGE_SIZE) -> Image.Image:
    """
    Resize image to have a maximum dimension of max_dim.
    Default max_dim for the Llama3.2 is 1120.
    """
    original_width, original_height = img.size

    if original_width > original_height:
        scale_factor = max_dim / original_width
    else:
        scale_factor = max_dim / original_height

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    return img.resize((new_width, new_height))


def process_image(image_path: str | None = None, image=None) -> Image.Image:
    """Process and validate image input"""
    if image is not None:
        return resize_image(image.convert("RGB"))
    if image_path and os.path.exists(image_path):
        return resize_image(Image.open(image_path).convert("RGB"))
    raise ValueError("No valid image provided")


def generate_text_from_image(
    model, processor, image, prompt_text: str, temperature: float, top_p: float
):
    """Generate text from image using model"""
    try:
        # Clear memory before generation
        torch.cuda.empty_cache()
        gc.collect()

        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
            }
        ]
        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Process inputs with memory optimization
        # Using the updated autocast API
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            inputs = processor(image, prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            output = model.generate(
                **inputs,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=MAX_OUTPUT_TOKENS,
                do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                num_beams=1,
            )

        # Clear memory after generation
        torch.cuda.empty_cache()
        gc.collect()

        return processor.decode(output[0], skip_special_tokens=True)

    except RuntimeError as e:
        print(f"Generation error: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        raise


def main(args):
    """Main execution flow"""
    try:
        torch.cuda.empty_cache()
        gc.collect()

        model, processor = load_model_and_processor(args.model_name)

        image = None
        if args.image_path:
            image = process_image(image_path=args.image_path)

        result = generate_text_from_image(
            model, processor, image, args.prompt_text, args.temperature, args.top_p
        )
        print("Generated Text:", result)

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        torch.cuda.empty_cache()
        gc.collect()
        raise

    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTX 4080 optimized Llama inference")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--prompt_text", type=str, help="Prompt text for the image")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.5, help="Top-p sampling")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="Model name")

    args = parser.parse_args()
    main(args)
