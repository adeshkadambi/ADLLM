# mypy: ignore-errors

import math
import json
import base64
import logging
from io import BytesIO

import ollama
from PIL import Image

from classes import ADL_DESCRIPTIONS


class ADLClassifier:

    def __init__(self, model: str = "llama3.2-vision:latest", **kwargs) -> None:
        self.model = model
        ollama.pull(self.model)

        self.system_prompt = """
        You are a highly experienced rehabilitation specialist with expertise in ADL classification.
        You have extensive experience working with stroke and SCI patients and understand how
        these conditions affect activity performance.
        """

        # set default options
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("seed", 101)

        self.kwargs = kwargs

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _create_frame_analysis_prompt(frame_number: int, total_frames: int) -> str:
        """Create prompt for analyzing individual frames."""

        prompt = f"""
        You are analyzing frame {frame_number} of {total_frames} from a first-person perspective video.
        Describe ONLY what you can directly observe in this single frame, without any interpretation:

        1. Static Elements
        - Room/location visible
        - Furniture or fixtures present
        - Objects visible in frame

        2. Person's Position
        - Hand position
        - Any object currently in hand
        - Body position (if visible)

        3. State of Objects
        - Position of objects relative to person
        - Whether objects are being actively manipulated

        Important: 
        - Do not make assumptions about activities
        - Do not interpret purpose of objects
        - Do not connect observations between frames
        - Do not speculate about intent

        Describe only what exists in this exact frame:
        """
        return prompt

    @staticmethod
    def _create_frame_context_synthesis(frame_descriptions: list[str]) -> str:
        """After getting individual frame descriptions, synthesize key patterns."""

        prompt = f"""
        **Review these {len(frame_descriptions)} frame descriptions and list:**

        1. Constant Elements
        - What objects remain in the same position across frames
        - What environment features are consistent

        2. Changes Between Frames
        - What objects change position
        - What hand positions change
        - What new objects appear or disappear

        Important: Only describe patterns that are explicitly mentioned in multiple frame descriptions.
        Do not interpret these patterns yet.

        **FRAME DESCRIPTIONS:**
        {frame_descriptions}
        """
        return prompt

    @staticmethod
    def _create_adl_classification_prompt(
        frame_descriptions: list[str], context_synthesis: str
    ) -> str:
        """Create prompt for classifying ADL based on frame descriptions and context synthesis."""
        prompt = f"""
        **Problem Statement**: 
        You are classifying an activity from the following options based on these observations and image grid. The image grid
        shows {len(frame_descriptions)} frames that were uniformly sampled from a video and arranged from top left to bottom right.

        **Activity Options**:
        {ADL_DESCRIPTIONS}

        **Frame Observations**:
        {frame_descriptions}

        **Temporal Analysis**:
        {context_synthesis}

        What activity is being performed in this video?

        **Solution Structure**:
        1. Begin by listing ALL observable evidence across frames:
        - Sustained actions/positions
        - Object interactions
        - Environmental context
        - Changes or lack of changes between frames

        2. Compare evidence against EACH activity category's required criteria:
        - Must check against ALL categories
        - Document which criteria are met AND not met for each category
        - Cannot skip categories even if one seems obvious
        - When in doubt, classify based on confirmed actions, not potential actions
        - Presence of objects alone does not indicate their use
        - Document why each unselected category does not fit

        Step 3: Confirm with expert opinion:
        - Consult with 5 occupational therapists and reach a consensus on the ADL classification
        - Document key points of discussion and reasoning

        Step 4: Evaluate consensus classification:
        - Have 3 different occupational therapists critically evaluate the classification
        - Document their findings and any disagreements

        Step 5: Verify final classification:
        - Confirm the final classification aligns with all evidence
        - Ensure the chosen category is one of the provided options

        **Required Response Format**:
        Respond with a valid JSON object using this exact structure:
        {{
            "ADL": "one of FEEDING, FUNCTIONAL MOBILITY, GROOMING AND HEALTH MANAGEMENT, COMMUNICATION MANAGEMENT, HOME MANAGEMENT, MEAL PREPARATION AND CLEANUP, or LEISURE",
            "Reasoning": "string containing detailed reasoning with specific frame references",
            "Activities": "string containing the sequence of actions observed",
            "Tags": "list of tags describing key actions and active objects to support classification",
            "Intermediate_Steps": {{
                "Environment_Analysis": "string describing step 1 findings",
                "ADL_Comparison": "string describing step 2 findings",
                "OT_Discussion": "string describing step 3 findings",
                "Expert_Evaluation": "string describing step 4 findings",
                "Final_Verification": "string describing step 5 findings"
            }}
        }}
        """
        return prompt

    def _get_model_response(self, prompt: str, image_base64: str | None = None) -> str:

        if image_base64 is None:
            message = {"role": "user", "content": prompt}
        else:
            message = {"role": "user", "content": prompt, "images": [image_base64]}

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                message,
            ],
            options=self.kwargs,
        )
        return response["message"]["content"]

    def _json_model_response(
        self, prompt: str, image_base64: str | None = None
    ) -> dict:
        if image_base64 is None:
            message = {"role": "user", "content": prompt}
        else:
            message = {"role": "user", "content": prompt, "images": [image_base64]}

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                message,
            ],
            options=self.kwargs,
            format="json",
        )
        return response["message"]["content"]

    def _resize_image(self, img: Image.Image, max_dim: int = 1120):
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

    def _encode_image(self, img: Image.Image) -> str:
        """Convert image to base64 string."""
        # Resize image to have a maximum dimension of 1120
        img_resized = self._resize_image(img, max_dim=1120)

        # Convert image to bytes
        buffered = BytesIO()
        img_resized.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        # Encode image to base64 string
        return base64.b64encode(img_bytes).decode("utf-8")

    def _create_image_grid(
        self, sampled_frames: list[Image.Image], max_dim: int = 1120
    ) -> Image.Image:
        """
        Create a grid of images from a list of PIL Images, maintaining aspect ratios where possible
        and ensuring no dimension exceeds max_dim.
        """
        # grid dimensions
        n_images = len(sampled_frames)
        n_cols = math.ceil(math.sqrt(n_images))
        n_rows = math.ceil(n_images / n_cols)

        # max width and height for each image
        max_cell_width = max_dim // n_cols
        max_cell_height = max_dim // n_rows

        # resize images
        resized_images = []

        for img in sampled_frames:
            aspect_ratio = img.width / img.height

            if aspect_ratio > 1:  # Wider than tall
                new_width = min(max_cell_width, int(max_cell_height * aspect_ratio))
                new_height = int(new_width / aspect_ratio)
            else:  # Taller than wide or square
                new_height = min(max_cell_height, int(max_cell_width / aspect_ratio))
                new_width = int(new_height * aspect_ratio)

            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_images.append(resized_img)

        # final grid size
        cell_width = max(img.width for img in resized_images)
        cell_height = max(img.height for img in resized_images)
        grid_width = cell_width * n_cols
        grid_height = cell_height * n_rows

        # create grid
        grid_image = Image.new("RGB", (grid_width, grid_height), color="white")

        for idx, img in enumerate(resized_images):
            row = idx // n_cols
            col = idx % n_cols

            # find center of each cell to paste image
            x = col * cell_width + (cell_width - img.width) // 2
            y = row * cell_height + (cell_height - img.height) // 2

            grid_image.paste(img, (x, y))

        # resize grid if necessary
        if grid_width > max_dim or grid_height > max_dim:
            aspect_ratio = grid_width / grid_height
            if grid_width > grid_height:
                new_width = max_dim
                new_height = int(max_dim / aspect_ratio)
            else:
                new_height = max_dim
                new_width = int(max_dim * aspect_ratio)
            grid_image = grid_image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )

        return grid_image

    def analyse_frames(
        self,
        sampled_frames: list[Image.Image],
        sampled_indices: list[int],
        total_frames: int,
    ) -> list[str]:
        """Get individual frame descriptions from sampled frames."""

        frame_descriptions = []
        for frame in sampled_frames:
            frame_base64 = self._encode_image(frame)
            frame_number = sampled_indices[sampled_frames.index(frame)]
            frame_prompt = self._create_frame_analysis_prompt(
                frame_number, total_frames
            )

            response = self._get_model_response(frame_prompt, frame_base64)
            frame_descriptions.append(response)

        self.logger.info("Frame analysis completed.")
        return frame_descriptions

    def synthesize_context(self, frame_descriptions: list[str]) -> str:
        """Synthesize context from list of individual frame descriptions."""

        synthesis_prompt = self._create_frame_context_synthesis(frame_descriptions)
        synthesis_response = self._get_model_response(synthesis_prompt)

        self.logger.info("Context synthesis completed.")
        return synthesis_response

    def classify_adl(
        self,
        frame_descriptions: list[str],
        context_synthesis: str,
        image_grid: Image.Image,
    ) -> str:
        """Classify ADL based on frame descriptions and context synthesis."""

        grid_base64 = self._encode_image(image_grid)

        classification_prompt = self._create_adl_classification_prompt(
            frame_descriptions, context_synthesis
        )
        classification_response = self._json_model_response(
            classification_prompt, grid_base64
        )

        self.logger.info("ADL classification completed.")
        return classification_response

    def predict(
        self,
        sampled_frames: list[Image.Image],
        sampled_indices: list[int],
        total_frames: int,
    ) -> dict:
        """
        Predict ADL based on sampled frames.
        """
        image_grid = self._create_image_grid(sampled_frames)

        # get frame descriptions, synthesize context, and classify ADL
        frame_descriptions = self.analyse_frames(
            sampled_frames, sampled_indices, total_frames
        )
        context_synthesis = self.synthesize_context(frame_descriptions)
        adl_classification = self.classify_adl(
            frame_descriptions, context_synthesis, image_grid
        )

        # parse response
        try:
            adl_classification = json.loads(adl_classification)
            adl_classification["Image_Grid"] = image_grid
            return adl_classification

        except json.JSONDecodeError:
            self.logger.error("Error parsing JSON response: %s", adl_classification)
            return {
                "error": "Error parsing JSON response. Please try again.",
                "Image_Grid": image_grid,
            }
