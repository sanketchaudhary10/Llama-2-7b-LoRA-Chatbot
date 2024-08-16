import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

model = "klyang/MentaLLaMA-chat-7B"

tokenizer = LlamaTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    device_map="auto",
)

# List of test prompts
test_prompts = [
    "I'm planning to have baby, so I have to quit smoking - but it's hard. Sometimes it's not a physical need, it's mental. I cannot help myself from thinking about smoking. What can I do to get rid of this addiction?",
    "I'm having issues with my relative. the police never believe the experiences i have been through because i am only a kid. i've even had trouble trying to reach a therapist because i said i wanted to get an adult to help me. could you please give me advice?",
    "I had a head injury a few years ago and my mind races all the time. I have trouble sleeping and have a lot of anxiety. Every medicine I have been on my body rejects; I get sick to my stomach and get blisters in my mouth. How can I calm my self down? I'm a wreck.",
    "My wife and mother are having tense disagreements. in the past, they have had minor differences. for example, my wife would complain to me my mother is too overbearing; my mother would complain my wife is lazy. however, it has intensified lately. I think the cause is my wife talked back to her once. now, any little disagreement is magnified, leading to major disagreements. what can I do?",
    ]

for prompt in test_prompts:
    sequences = pipeline(
        prompt,
        max_length=1024,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    for seq in sequences:
        print(f"Prompt: {prompt}")
        print(f"Result: {seq['generated_text']}")
        print("-" * 50)
