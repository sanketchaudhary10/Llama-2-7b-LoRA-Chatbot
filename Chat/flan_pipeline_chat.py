from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, pipeline

model_id = 'google/flan-t5-small'

config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, config=config)

# Corrected the pipeline call
flan_pipeline = pipeline('text2text-generation',
                         model=model_id,
                         tokenizer=tokenizer,
                         max_length=1024  # Adjust this value as needed
                         )

# Function to chat with FLan
def chat_with_flan(prompt):
    # Use the pipeline to generate a response based on the provided prompt
    response = flan_pipeline(prompt, max_length=1024) 

    # Extract the generated response from the pipeline output
    generated_response = response[0]['generated_text'].strip()

    return generated_response

# Example conversation
user_input = "I am feeling very depressed and I dont feel like talking to anyone. Sometimes I spend most of my time sitting alone in the room not talking to anybody for hours. Can you help me?"
flan_response = chat_with_flan(user_input)
print("User: {}".format(user_input))
print("Flan: {}".format(flan_response))
