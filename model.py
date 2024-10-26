# mypy: ignore-errors

import base64
from io import BytesIO
from dataclasses import dataclass

import ollama
from PIL import Image


@dataclass
class ADLClassifier:
    model: str = "x/llama3.2-vision"
    prompt: str = """
    Problem Statement: 
    What ADL or iADL is being performed in this image of an individual with stroke or spinal cord injury from the following options:
    a. Feeding - in front of a plate of food and/or drink and in the process of eating.
    b. Functional Mobility - in-bed mobility, wheelchair mobility, and transfers (e.g., wheelchair, bed, car, shower, tub, toilet, chair, floor).
    c. Grooming and Health Management - activities related to taking medication from pill bottles, doing exercise, or grooming (e.g., brushing teeth, combing hair, shaving, applying makeup, washing face, washing hands, applying lotion, dressing, undressing).
    d. Communication Management - activities related to using the telephone or smartphone, computer, writing, or other communication devices.
    e. Home Management - activities related to cleaning, laundry, or other household chores.
    f. Meal Preparation and Cleanup - Preparing a meal (often in the kitchen) or cleaning up after a meal (e.g., washing dishes, putting away food).
    g. Leisure and Other - activities that do not fit into the other categories.

    Solution Structure:
    1. Begin the response by considering the environment, visible objects and their usage, and hand positions or interactions.
    2. Based on the context from step 1, describe what the person *could* be doing. Genenate a list of possible activities.
    3. Debate the answer in a room of 5 occupational therapists. 4 out of 5 therapists should agree on the activity being performed.
    4. In new room of 3 occupational therapists, scrutinize the evidence and reasoning for the activity being performed. If they conclude previous response was incorrect, go back to step 2 and try again.
    5. Make sure the answer is one of the provided options. If not, go back to step 2.
    6. Finally, state "The ADL/iADL being performed in the image is [final answer] because [reasoning]."
    """

    def __post_init__(self):
        ollama.pull(self.model)

    def resize_image(self, img: Image.Image, max_dim: int = 1120):
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

    def encode_image(self, img: Image.Image) -> str:
        """Convert image to base64 string."""
        # Resize image to have a maximum dimension of 1120
        img_resized = self.resize_image(img, max_dim=1120)

        # Convert image to bytes
        buffered = BytesIO()
        img_resized.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        # Encode image to base64 string
        return base64.b64encode(img_bytes).decode("utf-8")

    def encode_image_from_path(self, image_path: str) -> str:
        """Convert image to base64 string."""
        with Image.open(image_path) as img:
            # Resize image to have a maximum dimension of 1120
            img_resized = self.resize_image(img, max_dim=1120)

            # Convert image to bytes
            buffered = BytesIO()
            img_resized.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            # Encode image to base64 string
            return base64.b64encode(img_bytes).decode("utf-8")

    def analyze_frame(self, image_path: str, question: str | None = None, **kwargs):
        """Analyze frame and return ADL class."""

        if question is None:
            question = self.prompt

        # set default options
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("seed", 101)

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": question,
                    "images": [self.encode_image(image_path)],
                },
            ],
            options=kwargs,
        )

        return response["message"]["content"]
