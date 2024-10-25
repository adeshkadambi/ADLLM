import base64
import requests
from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image
import ollama


def disp_image(address):
    if address.startswith("http://") or address.startswith("https://"):
        response = requests.get(address)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(address)

    plt.imshow(img)
    plt.axis("off")
    plt.show()


def encode_image(image_path: str) -> str:
    """Convert image to base64 string."""
    with Image.open(image_path) as img:
        # Resize image to have a maximum dimension of 1120
        img_resized = resize_image(img, max_dim=1120)

        # Convert image to bytes
        buffered = BytesIO()
        img_resized.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        # Encode image to base64 string
        return base64.b64encode(img_bytes).decode("utf-8")


def resize_image(img: Image.Image, max_dim: int = 1120):
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


def llama_response(path: str, prompt: str):
    ollama.pull("x/llama3.2-vision")
    return ollama.chat(
        model="x/llama3.2-vision",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [encode_image(path)],
            },
        ],
        options={
            "temperature": 0,
            "seed": 101,
        },
    )
