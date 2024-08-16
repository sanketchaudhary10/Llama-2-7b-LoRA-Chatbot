from transformers import T5ForConditionalGeneration, AutoTokenizer

model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)                                                                 
tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

input_string = "I am feeling quite depressed and don't feel like talking to anyone, what can i do"                                               

# Move input tensor to GPU
inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")

# Generate text on the GPU
outputs = model.generate(inputs, max_length=2048)

# Decode and print the generated output
print(tokenizer.decode(outputs[0]))