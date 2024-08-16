import transformers

# Load model and tokenizer
model = transformers.AutoModelForCausalLM.from_pretrained("/N/project/GenAIMH/llama2_7b_chat/Llama-2_7B-chat_50/checkpoint-5200") 
tokenizer = transformers.AutoTokenizer.from_pretrained("/N/project/GenAIMH/llama2_7b_chat/Llama-2_7B-chat_3/checkpoint-30")

# Bot starter sentence  
bot_starter = "Hi there! How are you feeling today?"

# Sample conversation 
conversation = f"{bot_starter}\n"  

while True:
    # Get user input
    user_input = input("You: ")
    
    # Add user input to conversation
    conversation += f"{user_input}\n"

    # Encode conversation 
    inputs = tokenizer([conversation], return_tensors="pt").input_ids
    
    # Generate response     
    bot_response = model.generate(inputs)[0] 
    
    # Print response
    print(f"Bot: {tokenizer.decode(bot_response)}")
    
    conversation += f"{tokenizer.decode(bot_response)}\n"