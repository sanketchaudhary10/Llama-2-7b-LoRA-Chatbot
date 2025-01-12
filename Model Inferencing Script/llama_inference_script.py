import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import BitsAndBytesConfig

# ------------------------------
# Configuration
# ------------------------------

# Paths and identifiers
BASE_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Base model name
PEFT_MODEL_PATH = "/N/project/GenAIMH/llama2_7b_chat/Llama-2_7B-chat_3"     # Path to your fine-tuned model

# Hugging Face authentication token
HF_TOKEN = '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'  # Replace with your actual token

# System message used during training
SYSTEM_MESSAGE = "Hello, this is an automated response. Please seek professional help if needed."

# Quantization configuration (consistent with training)
NF4_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16  # Adjust based on your GPU
)

# Generation parameters
MAX_LENGTH = 2048
TEMPERATURE = 0.7
TOP_P = 0.9

# ------------------------------
# Load Tokenizer
# ------------------------------

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    use_auth_token=HF_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ------------------------------
# Load Base Model
# ------------------------------

print("Loading base model with quantization...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=NF4_CONFIG,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=HF_TOKEN
)

# ------------------------------
# Load Fine-Tuned (LoRA) Model
# ------------------------------

print("Loading fine-tuned LoRA model...")
model = PeftModel.from_pretrained(
    base_model,
    PEFT_MODEL_PATH,
    device_map="auto"
)

# Ensure the model is in evaluation mode
model.eval()

# ------------------------------
# Define Prompt Formatting Function
# ------------------------------

def format_prompt(user_input: str) -> str:
    """
    Formats the user input to match the training data format.
    """
    return f"<s>[INST] <<SYS>>{SYSTEM_MESSAGE}<</SYS>> {user_input} [/INST]"

# ------------------------------
# Define Inference Function
# ------------------------------

def generate_response(user_input: str,
                      max_length: int = MAX_LENGTH,
                      temperature: float = TEMPERATURE,
                      top_p: float = TOP_P) -> str:
    """
    Generates a response from the fine-tuned model based on the user input.
    """
    # Format the input prompt
    formatted_prompt = format_prompt(user_input)
    
    # Tokenize the input
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the generated tokens
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated response by removing the input prompt
    response = output_text[len(formatted_prompt):].strip()
    
    return response

# ------------------------------
# Main Inference Function
# ------------------------------

def main():
    print("Inference setup complete. You can start interacting with the model.")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting inference loop.")
            break
        response = generate_response(user_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
