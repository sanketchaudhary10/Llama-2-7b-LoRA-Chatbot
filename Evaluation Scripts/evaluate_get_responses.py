# Original Script
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained("/N/project/GenAIMH/llama2_7b_chat/Llama-2_7B-chat_10/checkpoint-2080")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("/N/project/GenAIMH/llama2_7b_chat/Llama-2_7B-chat_10/checkpoint-2080")

# Example entries for the test dataset
test_entries = [
    {"text": "### Human: I'm feeling really anxious about an upcoming event. I don't know how to handle it."},
    {"text": "### Human: I've been struggling with low self-esteem lately. It's affecting my daily life."},
    {"text": "### Human: I lost my job recently, and I'm feeling overwhelmed."},
    {"text": "### Human: I'm struggling with insomnia and can't seem to get a good night's sleep."},
    {"text": "### Human: Dealing with family conflicts is stressing me out. What should I do?"},
]

# Corresponding reference responses
reference_responses = [
    "Assistant: I'm sorry to hear that you're feeling anxious. It's completely normal to feel this way before an important event. Take a deep breath and try to break down the tasks into smaller, more manageable steps. Remember, I'm here to support you.",
    "Assistant: I'm sorry to hear that you're going through a tough time. Remember, you are worthy and valuable just as you are. It might be helpful to focus on your strengths and accomplishments. If you feel comfortable, consider talking to a friend or seeking professional support to explore these feelings further.",
    "Assistant: I'm really sorry to hear about your job loss. It's completely normal to feel overwhelmed during such challenging times. Take things one step at a time, and consider reaching out to your support network. If you need help exploring new opportunities or coping strategies, I'm here to assist you.",
    "Assistant: I'm sorry to hear that you're having trouble sleeping. Establishing a bedtime routine and creating a relaxing environment can be helpful. It might also be beneficial to limit screen time before bed and consider relaxation techniques like deep breathing. If the issue persists, consulting with a healthcare professional could provide additional support.",
    "Assistant: Family conflicts can be challenging. It might be helpful to express your feelings openly, but also listen to the perspectives of others. If communication is difficult, a mediator or counselor could assist in facilitating a constructive conversation. Remember, taking care of your mental well-being is important, and I'm here to help you navigate through this."
]

# Generate responses
generated_responses = []

for entry in test_entries:
    # Tokenize input
    input_text = entry["text"]
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=150, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Decode and extract assistant's response
    generated_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assistant_response = generated_response[len(input_text):].strip()  

    generated_responses.append(assistant_response)

# Print generated responses
for entry, response in zip(test_entries, generated_responses):
    print(f"Input: {entry['text']}")
    print(f"Generated Response: {response}\n")



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')

# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained("/N/project/GenAIMH/llama2_7b_chat/Llama-2_7B-chat_10/checkpoint-2080")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("/N/project/GenAIMH/llama2_7b_chat/Llama-2_7B-chat_10/checkpoint-2080")

# Example entries for the test dataset
test_entries = [
    {"text": "### Human: I'm feeling really anxious about an upcoming event. I don't know how to handle it."},
    {"text": "### Human: I've been struggling with low self-esteem lately. It's affecting my daily life."},
    {"text": "### Human: I lost my job recently, and I'm feeling overwhelmed."},
    {"text": "### Human: I'm struggling with insomnia and can't seem to get a good night's sleep."},
    {"text": "### Human: Dealing with family conflicts is stressing me out. What should I do?"},
]

# Corresponding reference responses
reference_responses = [
    "Assistant: I'm sorry to hear that you're feeling anxious. It's completely normal to feel this way before an important event. Take a deep breath and try to break down the tasks into smaller, more manageable steps. Remember, I'm here to support you.",
    "Assistant: I'm sorry to hear that you're going through a tough time. Remember, you are worthy and valuable just as you are. It might be helpful to focus on your strengths and accomplishments. If you feel comfortable, consider talking to a friend or seeking professional support to explore these feelings further.",
    "Assistant: I'm really sorry to hear about your job loss. It's completely normal to feel overwhelmed during such challenging times. Take things one step at a time, and consider reaching out to your support network. If you need help exploring new opportunities or coping strategies, I'm here to assist you.",
    "Assistant: I'm sorry to hear that you're having trouble sleeping. Establishing a bedtime routine and creating a relaxing environment can be helpful. It might also be beneficial to limit screen time before bed and consider relaxation techniques like deep breathing. If the issue persists, consulting with a healthcare professional could provide additional support.",
    "Assistant: Family conflicts can be challenging. It might be helpful to express your feelings openly, but also listen to the perspectives of others. If communication is difficult, a mediator or counselor could assist in facilitating a constructive conversation. Remember, taking care of your mental well-being is important, and I'm here to help you navigate through this."
]

# Generate responses
generated_responses = []

for entry in test_entries:
    # Tokenize input
    input_text = entry["text"]
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=150, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Decode and extract assistant's response
    generated_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assistant_response = generated_response[len(input_text):].strip()

    generated_responses.append(assistant_response)

# Print generated responses
for entry, response in zip(test_entries, generated_responses):
    print(f"Input: {entry['text']}")
    print(f"Generated Response: {response}")

# Calculate METEOR score
meteor_score_value = meteor_score(reference_responses, generated_responses)
print(f"\nMETEOR Score: {meteor_score_value}")
