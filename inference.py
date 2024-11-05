# mypy: ignore-errors

import base64
import math
from io import BytesIO

import ollama
from PIL import Image
from tqdm import tqdm

from classes import ADL_DESCRIPTIONS


class ADLClassifier:

    def __init__(self, model: str = "x/llama3.2-vision", **kwargs) -> None:
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

    @staticmethod
    def _create_frame_analysis_prompt(frame_number: int, total_frames: int) -> str:
        prompt = f"""
        You are analyzing frame {frame_number} of {total_frames} from a video showing
        an activity of daily living performed by someone with stroke or spinal cord injury.
        The video is recorded using a head-mounted GoPro camera and shows a first-person perspective.

        Provide a detailed observation focusing on:

        1. Environment
        - Identify the room or setting (e.g., kitchen, bathroom, bedroom)
        - Note relevant fixtures or furniture
        - Describe lighting and general layout

        2. Objects and Equipment
        - List visible objects
        - Note any assistive devices or medical equipment
        - Identify objects being actively used or manipulated

        3. Hand and Body Position
        - Describe hand placement and any object interactions
        - Note grip patterns or types
        - Describe body positioning relative to objects/surfaces
        - Identify any compensatory movements or strategies

        4. Movements and Actions
        - Describe any active movements or actions

        Format your response as a clear, objective description without making assumptions about the overall activity. Focus on what you can directly observe in this specific frame.

        Provide your observation:
        """
        return prompt

    @staticmethod
    def _create_frame_context_synthesis(frame_descriptions: list[str]) -> str:
        """
        After getting individual frame descriptions, synthesize key patterns.
        """
        prompt = f"""
        Review these detailed observations from {len(frame_descriptions)} first-person perspective video frames:

        {frame_descriptions}

        Provide a concise synthesis focusing on:
        1. Consistent elements (objects, environment) across frames
        2. Object interactions with hands across frames

        Focus only on describing patterns and sequences objectively.
        """
        return prompt

    @staticmethod
    def _create_adl_classification_prompt(
        frame_descriptions: list[str], context_synthesis: str
    ) -> str:
        """
        After synthesizing context, classify the ADL being performed.
        """
        prompt = f"""
        **Problem Statement**: 
        {len(frame_descriptions)} frames were sampled from a first-person perspective video of an
        individual with stroke or spinal cord injury. The provided image grid shows the frames in 
        sequence from top left to bottom right.

        **Temporal Analysis**:
        Here is a sythesis of observations from the frames:

        {context_synthesis}

        What ADL or iADL is being performed in this video? 
        
        Consider these options:

        {ADL_DESCRIPTIONS}

        **Solution Structure**:
        1. Begin the response by considering the environment, visible objects and their usage. 
        What environment is shown across frames? What key objects appear consistently? 
        How are objects being used or manipulated? What interactions are observed?

        2. Based on the context from Step 1 and the image grid, what sequence of actions is visible across frames?
        Generate candidate ADLs or iADLs from the provided list based on these actions.

        3. Simulate a discussion among 5 occupational therapists to reach a consensus on the ADL classification:
            - Consider temporal progression across frames
            - Evaluate evidence for each possible ADL
            - Require 4/5 therapists to agree on classification
            - Document key points of discussion

        4. Have 3 different OTs critically evaluate the consensus classification by challenging assumptions from 
        initial classification, consider alternative interpretations, and verify temporal consistency. Document their findings.
        If they disagree with the consensus classification, return to step 2.

        5. Confirm final classification matches one of the provided options. If not, return to step 2. Ensure all evidence aligns with chosen category.

        **Required Response Format**:
        You must respond with a valid Python dictionary object using this exact structure:
        {{{{
            "ADL": "string containing final classification",
            "Reasoning": "string containing detailed reasoning with specific frame references",
            "Intermediate_Steps": {{{{
                "Environment_Analysis": "string describing step 1 findings",
                "Action_Sequence": "string describing step 2 findings",
                "OT_Discussion": "string describing step 3 findings",
                "Critical_Evaluation": "string describing step 4 findings",
                "Final_Verification": "string describing step 5 findings"
            }}}}
        }}}}

        IMPORTANT: Your entire response must be a valid Python dictionary object. Do not include any text outside the Python dictionary structure.
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
        for frame in tqdm(sampled_frames, total=len(sampled_frames)):
            frame_base64 = self._encode_image(frame)
            frame_number = sampled_indices[sampled_frames.index(frame)]
            frame_prompt = self._create_frame_analysis_prompt(
                frame_number, total_frames
            )

            response = self._get_model_response(frame_prompt, frame_base64)
            frame_descriptions.append(response)

        print("Frame analysis completed.")
        return frame_descriptions

    def synthesize_context(self, frame_descriptions: list[str]) -> str:
        """Synthesize context from list of individual frame descriptions."""

        synthesis_prompt = self._create_frame_context_synthesis(frame_descriptions)
        synthesis_response = self._get_model_response(synthesis_prompt)

        print("Frame synthesis completed.")
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
        classification_response = self._get_model_response(
            classification_prompt, grid_base64
        )

        print("ADL classification completed.")
        return classification_response

    def predict(
        self,
        sampled_frames: list[Image.Image],
        sampled_indices: list[int],
        total_frames: int,
    ) -> tuple[str, Image.Image]:
        """
        Predict ADL based on sampled frames.
        """
        image_grid = self._create_image_grid(sampled_frames)

        frame_descriptions = self.analyse_frames(
            sampled_frames, sampled_indices, total_frames
        )
        context_synthesis = self.synthesize_context(frame_descriptions)
        adl_classification = self.classify_adl(
            frame_descriptions, context_synthesis, image_grid
        )
        return adl_classification, image_grid
