from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "mental/mental-roberta-base"

cache_dir_name = "N/slate/chaudhs/cache_roberta_model"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    cache_dir=cache_dir_name,
    local_dir_use_symlinks=False
)

# prompts to test the model
test_prompts = [
    "I'm in a state of depression right now. Who can I talk to? I've been sick in a lot of pain and crying. Don't know where to turn.",
    "i have been diagnosed with adhd and experienced manic depression episodes.  i have problems with anger management.  apparently, i also have an odd, bipolar and split personality.  how can i be truly happy?",
    "I'm planning to have baby, so I have to quit smoking - but it's hard. Sometimes it's not a physical need, it's mental. I cannot help myself from thinking about smoking. What can I do to get rid of this addiction?",
    "I had a head injury a few years ago and my mind races all the time. I have trouble sleeping and have a lot of anxiety. Every medicine I have been on my body rejects; I get sick to my stomach and get blisters in my mouth. How can I calm my self down? I'm a wreck.",
    "I'm having issues with my relative. the police never believe the experiences i have been through because i am only a kid. i've even had trouble trying to reach a therapist because i said i wanted to get an adult to help me. could you please give me advice?"
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
        print(f"Prompt (Mental-RoBERTA): {prompt}")
        print(f"Result (Mental-RoBERTA): {seq['generated_text']}")
        print("-" * 50)

