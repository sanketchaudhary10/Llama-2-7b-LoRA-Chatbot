import transformers
import torch
from transformers import AutoTokenizer, pipeline

def choose_model():
    print("Choose a model:")
    print("1. Mental Flan")
    print("2. Mental Alpaca")
    print("3. Mental Llama")
    print("4. Mental RoBERTA")
    print("5. Falcon")
    choice = input("Enter the number corresponding to the model you want to use: ")
    return choice

def load_model(choice):
    models = {
        "1": "NEU-HAI/mental-flan-t5-xxl",
        "2": "NEU-HAI/Llama-2-7b-alpaca-cleaned",
        "3": "klyang/MentaLLaMA-chat-7B",
        "4": "mental/mental-roberta-base",
        "5": "tiiuae/falcon-7b-instruct"
    }
    model_name = models.get(choice)
    if model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        chat_pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        return chat_pipeline
    else:
        print("Invalid choice. Please choose a number between 1 and 5.")

def chat_with_model(chat_pipeline):
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting chat.")
            break
        response = chat_pipeline(user_input, max_length=1024, do_sample=True, top_k=10, num_return_sequences=1)
        print("Model:", response[0]['generated_text'])

def main():
    choice = choose_model()
    chat_pipeline = load_model(choice)
    if chat_pipeline:
        print("Type 'exit' or 'quit' to end the conversation.")
        chat_with_model(chat_pipeline)

if __name__ == "__main__":
    main()
