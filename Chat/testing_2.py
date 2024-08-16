print("Llama 2 7B Chat: Testing")
print("Begin:")


from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the path to the fine-tuned model
refined_model_path = "/N/project/GenAIMH/llama2_7b_chat/results_modified2_5/checkpoint-2600"

# Load the fine-tuned model and tokenizer
refined_model = AutoModelForCausalLM.from_pretrained(refined_model_path)
tokenizer = AutoTokenizer.from_pretrained(refined_model_path)

# Function to generate text for a given input
def generate_text(input_text):
    input_prompt = f"<s>[INST] {input_text} [/INST]"
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt")

    # Generate text using the fine-tuned model
    output = refined_model.generate(input_ids, max_length=200, num_beams=5, temperature=0.8, no_repeat_ngram_size=2)

    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example input
input_text = "I feel curious about the world and want to learn more about it."

# Generate text for the input
generated_output = generate_text(input_text)

# Print the generated output
print(generated_output)
