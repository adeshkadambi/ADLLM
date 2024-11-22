from classes import ADL_DESCRIPTIONS


def create_system_prompt() -> str:
    """Create prompt for the AI system's role and responsibilities."""

    prompt_2024_11_11 = """
    You are a highly experienced rehabilitation specialist with expertise in ADL classification.
    You have extensive experience working with stroke and SCI patients and understand how
    these conditions affect activity performance.
    """

    prompt = """
    You are an expert occupational therapy AI assistant tasked with analyzing egocentric (first-person) video data
    of patients with hand impairments performing unconstrained activities of daily living (ADLs) at home. Your role
    is to make objective observations based on the visual data from the video frames. You may be asked to analyze 
    individual frames or a sequence of frames arranged in a grid from top left to bottom right.

    Remember: your observations will support clinical decision-making and patient care. Maintain objectivity and
    a high standard for evidence-based reasoning in your responses. Clearly document your observations and rationale.
    """

    return prompt


def create_frame_analysis_prompt(frame_number: int, total_frames: int) -> str:
    """Create prompt for analyzing individual frames."""

    prompt_2024_11_11 = f"""
    You are analyzing frame {frame_number} of {total_frames} from a first-person
    perspective video captured using a head-mounted GoPro camera. Describe ONLY what you
    can directly observe in this single frame, without any interpretation:

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

    Describe only what exists in this exact frame by placing more emphasis on objects
    manipulated by the person:
    """

    prompt = f"""
    This is frame {frame_number} of {total_frames} from a 1-minute video captured using
    a head-mounted GoPro camera of a person performing an activity at home. Describe
    what you see in this frame, focusing on: the room or location, objects present,
    objects being actively interacted with, and the person's actions.

    Remember to be objective and avoid making assumptions.
    """

    return prompt


def create_frame_context_synthesis(frame_descriptions: list[str]) -> str:
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


def create_adl_classification_prompt(
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


def adl_classification_2024_11_17(frame_descriptions: list[str]) -> str:
    """Create prompt for classifying ADL based on frame descriptions and context synthesis."""
    prompt = f"""
    **Problem Statement**:
    Using the provided descriptions of {len(frame_descriptions)} frames uniformly sampled from a 1-minute video
    as well as a grid showing these frames, generate a list of tags that describe what is happening in the video.
    The tags should be based on the observable actions and objects in the frames. The purpose of these tags is to help
    occupational therapists quickly search for relevant videos based on the activities being performed.

    Ensure that the tags are specific, accurate, and concise. Avoid making assumptions or interpretations beyond what is
    directly observable in the video frames.

    **Frame Descriptions**:
    {frame_descriptions}

    **Solution Structure**:
    1. Initial Frame Analysis:
    - List ALL observable actions and objects
    VERIFY: Have you systematically reviewed each frame?

    2. Expert Tag Generation:
    Imagine three different occupational therapists are assigning tags to this video based on the observed actions and objects independently.
    All therapists will write down their tags and share them with the group. The tags that are common among all three therapists will be considered for the final list.

    **Tag Quality Criteria**:
    - Must be observable in at least 2 frames
    - Must be specific enough to distinguish between similar activities
    - Must avoid interpretation beyond visual evidence
    - Must use standardized terminology when applicable
    - Must be a single word or short phrase (e.g., 'eating', 'drinking', 'phone use', 'cooking', 'cleaning', 'reading', 
    'watching TV', 'writing', 'exercising', 'grooming', 'laundry', 'leisure', 'computer use', etc.)
    
    **Common Failure Modes to Avoid**:
    - Over-interpretation of single frames
    - Combining multiple distinct actions into one tag
    - Using overly broad or vague terms

    VERIFY: Does each remaining tag meet all criteria? Remove any tags that do not meet the criteria.

    3. Final Tag Selection:
    Compare the list of tags generated by the therapists in the previous step with the frames in the frame grid and select ONLY UP TO 3 tags that are most relevant to the video content.
    Provide a brief reasoning for each tag selected, explaining why it is relevant based on the observed actions and objects in the video frames.

    5. Activity Selection: 
    Based on the MAJORITY of frames and the generated tags, what activity do you think is being performed in the video from the following options:
    
    **Activity Options**:
    {ADL_DESCRIPTIONS}

    6. Response Validation:
    **Final Validation Checklist**:
    Before submitting response:
    1. Verify each tag appears in multiple frames
    2. Confirm tag specificity and clarity
    3. Check consistency between tags and selected activity
    4. Ensure reasoning references specific visual evidence

    **Required Response Format**:
    Respond with the primary activity from the options, a list of tags, and reasoning in JSON format. The structure should be as follows:
    {{
        "Activity": "one of FEEDING, FUNCTIONAL MOBILITY, GROOMING AND HEALTH MANAGEMENT, COMMUNICATION MANAGEMENT, HOME MANAGEMENT, MEAL PREPARATION AND CLEANUP, or LEISURE",
        "Alternate Activities": ["optional list of other possible activities from A to G"],
        "Final List of Tags": ["tag1", "tag2", "tag3", ...],
        "Reasoning": ["reasoning for tag1", "reasoning for tag2", "reasoning for tag3", ...]
    }}
    """
    return prompt


def adl_classification_2024_11_19(frame_descriptions: list[str]) -> str:
    """Create prompt for classifying ADL based on frame descriptions and context synthesis."""
    prompt = f"""
    **Problem Statement**:
    Using the provided descriptions of {len(frame_descriptions)} frames uniformly sampled from a 1-minute video
    as well as a grid showing these frames, generate a list of tags that describe what is happening in the video.
    The tags should be based on the observable actions and objects in the frames. The purpose of these tags is to help
    occupational therapists quickly search for relevant videos based on the activities being performed.

    Ensure that the tags are specific, accurate, and concise. Avoid making assumptions or interpretations beyond what is
    directly observable in the video frames.

    **Frame Descriptions**:
    {frame_descriptions}

    **Solution Structure**:
    1. Initial Frame Analysis:
    - List ALL observable actions and objects
    VERIFY: Have you systematically reviewed each frame?

    2. Observe Temporal Patterns:
    - Identify consistent actions or objects across frames
    VERIFY: Have you noted any patterns or changes between frames?

    3. Expert Tag Generation:
    Imagine three different occupational therapists are assigning tags to this video based on the observed actions and objects independently.
    All therapists will write down their tags and share them with the group. The tags that are common among all three therapists will be considered for the final list.

    **Tag Quality Criteria**:
    - Must be observable in at least 2 frames
    - Must be specific enough to distinguish between similar activities
    - Must avoid interpretation beyond visual evidence
    - Must use standardized terminology when applicable
    - Must be a single word or short phrase
    
    NOTE: Some examples of good tags include: 'eating', 'drinking', 'phone use', 'cooking', 'cleaning', 'reading', 'watching TV', 'writing', 'exercising', 
    'grooming', 'laundry', 'leisure', or 'computer use'. This is not an exhaustive list so feel free to add other relevant tags.
    
    **Common Failure Modes to Avoid**:
    - Over-interpretation of single frames
    - Combining multiple distinct actions into one tag
    - Using overly broad or vague terms

    VERIFY: Does each remaining tag meet all criteria? Remove any tags that do not meet the criteria.

    4. Final Tag Selection:
    Compare the list of tags generated by the therapists in the previous step with the frames in the frame grid and select ONLY UP TO 3 tags that are most relevant to the video content.
    Provide a brief reasoning for each tag selected, explaining why it is relevant based on the observed actions and objects in the video frames.

    5. Activity Selection: 
    Based on the MAJORITY of frames and the generated tags, what activity category do you think is being performed in the video from the following options:
    
    **Activity Category Options**:
    The following are activity categories. Each category has an overall definition from the occupational therapy handbook, and then a list of possible activities that fall under that category. 
    The activities are not exhaustive, but are meant to give a general idea of what activities fall under each category.

    A) FEEDING:: 
    Definition: 'Bringing food [or fluid] from the plate or cup to the mouth.'
    Sub-activities: eating, drinking

    B) FUNCTIONAL MOBILITY::
    Definition: 'Bed mobility, wheelchair use, and transfers.'
    Sub-activities: moving around using a walker, wheelchair, or power wheelchair. Transferring from bed to chair, chair to toilet, chair to chair, etc.

    C) GROOMING AND HEALTH MANAGEMENT::
    Definition: 'Self-care (washing, drying, combing, styling, brushing, and trimming hair; caring for nails (hands and feet); caring for skin, ears, eyes, and nose; applying deodorant; cleaning mouth; brushing and flossing teeth. Exercise and medication routines.'
    Sub-activities: brushing hair, washing face, brushing teeth, applying deodorant, taking medication, exercising, grooming

    D) COMMUNICATION MANAGEMENT::
    Definition: 'Using a variety of systems and equipment, including writing, telephones, and computers.'
    Sub-activities: using smartphones, using computers, writing
    IMPORTANT NOTE: If they are using the computer or phone, regardless of what they are doing on it, it is communication management.   

    E) HOME MANAGEMENT::
    Definition: 'Activities related to the maintenance of a household, including cleaning, laundry, and household chores.'
    Sub-activities: cleaning, laundry, household chores, organizing, gardening, garage work

    F) MEAL PREPARATION AND CLEANUP::
    Definition: 'Activities related to preparing and cleaning up after meals.'
    Sub-activities: cooking, washing dishes

    G) LEISURE::
    Definition: 'Nonobligatory activity that is not any of the aforementioned categories.'
    Sub-activities: watching TV, playing video games, reading, knitting, arts and crafts
    IMPORTANT NOTE: If they are on the computer or phone, regardless of what they are doing on it, it is communication management, NOT leisure.

    6. Response Validation:
    **Final Validation Checklist**:
    Before submitting response:
    1. Verify each tag appears in multiple frames
    2. Confirm tag specificity and clarity
    3. Check consistency between tags and selected activity
    4. Ensure reasoning references specific visual evidence

    IMPORTANT NOTE: If they are on the computer or phone, regardless of what they are doing on it, it is communication management, NOT leisure. Change the activity category if necessary.

    **Required Response Format**:
    Respond with the primary activity from the options, a list of tags, and reasoning in JSON format. The structure should be as follows:
    {{
        "Activity": "selected activity category from A to G, print the full name of the category here (e.g., 'FEEDING')",
        "Alternate Activities": ["optional list of other possible activity categories from A to G"],
        "Final List of Tags": ["tag1", "tag2", "tag3", ...],
        "Reasoning": ["reasoning for tag1", "reasoning for tag2", "reasoning for tag3", ...]
    }}
    """
    return prompt
