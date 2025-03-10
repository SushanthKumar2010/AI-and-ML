from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()
generator = pipeline("text-generation", model="gpt2")

class InputData(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "AI Text Generator API is running!"}

@app.post("/generate/")
def generate_text(data: InputData):
    output = generator(data.prompt, max_length=100, num_return_sequences=1)
    return {"generated_text": output[0]["generated_text"]}
