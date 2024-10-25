import ollama
import utils

ollama.pull("x/llama3.2-vision")

stream = ollama.chat(
    model="x/llama3.2-vision",
    messages=[
        {
            "role": "user",
            "content": "Can you tell me what an occupational therapist does in one sentence?",
            
        }
    ],
)

print(stream)
