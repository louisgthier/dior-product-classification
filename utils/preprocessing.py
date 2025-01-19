# RMBG-2.0 Setup
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import numpy as np
import io
import os
import hashlib

# Define the base cache directory.
BASE_CACHE_DIR = '.cache'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
print(f"Using device: {device}")

rmbg_2_model, rmbg_2_transform = None, None
rembg_session = None

def load_model(background_removal_method):
    """Load the RMBG-2.0 or rembg model."""
    global rmbg_2_model, rmbg_2_transform, rembg_session, remove
    if background_removal_method == 'RMBG_2':
        if not rmbg_2_model:
            rmbg_2_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
            torch.set_float32_matmul_precision('high')
            rmbg_2_model.to(device)
            rmbg_2_model.eval()

            image_size = (1024, 1024)
            rmbg_2_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        if not rembg_session:
            from rembg import remove, new_session
            rembg_session = new_session('u2net')

def compute_image_hash(image: Image.Image) -> str:
    """Compute a SHA256 hash for the given PIL image."""
    # Convert image to bytes in a consistent format (e.g., PNG) for hashing.
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
    return hashlib.sha256(image_bytes).hexdigest()

def compute_file_hash(file_path: str) -> str:
    """Compute a SHA256 hash for the given file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def remove_background(input_image: Image.Image, input_path: str = None, background_removal="RMBG_2") -> Image.Image:
    """
    Remove background from the input image using either rembg or RMBG-2.0
    based on the configuration flag, with caching based on image hash.
    """
    
    # Load the model if not already loaded.
    load_model(background_removal)
    
    # Determine the subdirectory based on the method.
    
    # Full cache directory for the current method.
    cache_dir = os.path.join(BASE_CACHE_DIR, background_removal)
    
    os.makedirs(cache_dir, exist_ok=True)

    # Compute hash for the input image.
    image_hash = compute_file_hash(input_path)
    # Use .png extension to support RGBA images.
    cached_filename = f"{image_hash}.png"
    cached_path = os.path.join(cache_dir, cached_filename)

    # Check if a cached version exists.
    if os.path.exists(cached_path):
        return Image.open(cached_path)

    # No cached image found; proceed with background removal.
    if background_removal == "RMBG_2":
        # Prepare input for RMBG-2.0
        input_img_resized = input_image.convert('RGB')
        input_tensor = rmbg_2_transform(input_img_resized).unsqueeze(0).to(device)

        # Prediction using RMBG-2.0
        with torch.no_grad():
            preds = rmbg_2_model(input_tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        # Convert prediction mask to PIL image
        pred_pil = transforms.ToPILImage()(pred)

        # Resize mask to original image size
        mask = pred_pil.resize(input_image.size)

        # Apply mask as alpha channel to original image
        image_rgba = input_image.convert("RGBA")
        image_rgba.putalpha(mask)
        result_image = image_rgba

    else:
        # Use rembg for background removal
        input_converted = input_image.convert('RGB')
        result = remove(input_converted, session=rembg_session)
        # rembg returns a bytes-like object or PIL image; ensure it's a PIL image.
        if isinstance(result, bytes):
            result_image = Image.open(io.BytesIO(result))
        else:
            result_image = result.convert("RGBA")

    # Save the processed image to the cache as PNG.
    result_image.save(cached_path, format="PNG")

    return result_image

def preprocess_image(input_path: str, background_removal="RMBG_2") -> Image.Image:
    """
    Preprocess the input image.
    """
    
    input: Image = Image.open(input_path)
    
    # if has alpha channel, use it directly; otherwise, remove background
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    if has_alpha:
        output = input
    else:
        input = input.convert('RGB')
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        output = remove_background(input, input_path, background_removal=background_removal)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    try:
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    except ValueError:
        return None
        
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    output = output.resize((518, 518), Image.Resampling.LANCZOS)
    output = np.array(output).astype(np.float32) / 255
    
    # Set every pixel with alpha less than 0.8 to (255, 255, 255)
    output[output[:, :, 3] < 0.8] = [1, 1, 1, 0]
    output = output[:, :, :3]
    
    # Remove the alpha channel
    # output = output[:, :, :3] * output[:, :, 3:4]
    
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output
