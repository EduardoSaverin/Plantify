from dotenv import load_dotenv
import os
import ollama

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")

def healthcheck():
    print(f"Connecting to {OLLAMA_HOST}")
    client = ollama.Client(host=OLLAMA_HOST)
    print(f"Connected to {OLLAMA_HOST}")
    models = client.list()
    for model in models['models']:
        name = model['model']
        details = client.show(name)
        capabilities = details.get('capabilities', [])
        model_type = "Vision" if "vision" in capabilities else "Text"
        print(f"{name} : {model_type}")


if __name__ == "__main__":
    healthcheck()
