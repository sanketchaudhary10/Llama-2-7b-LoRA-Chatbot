import random
import gc
import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import numpy as np
import pandas as pd
import transformers
import accelerate
import bitsandbytes as bnb
from datasets import load_dataset, concatenate_datasets
import torch


data_name = "formatted_conversations.jsonl"

data = []

with open(data_name, "r", encoding="utf-8") as file:
    for line in file:
        data.append({"text": line.strip()})

# Create a dictionary with a single column named "text"
data_dict = {"text": [item["text"] for item in data]}
# Create a Hugging Face Dataset
train_dataset = Dataset.from_dict(data_dict)


train_dataset


def format_llama(entry):
    system_message = "Hello, this is an automated response. Please seek professional help if needed. "
    formatted_entry = f"<s>[INST] <<SYS>>{system_message}<</SYS>>{entry['text']} [/INST]  {entry['text']}  </s>"
    return formatted_entry

formatted_entries = []  # To store the formatted entries

for entry in train_dataset['text']:
    formatted_entries.append({"text": format_llama({"text": entry})})


# Printing the first 5 formatted entries
for formatted_entry in formatted_entries[:5]:
    print(formatted_entry)


# Splitting the dataset into training and validation sets
train_size = int(len(formatted_entries) * 0.8)  # 80% for training
val_size = len(formatted_entries) - train_size  # Remaining 20% for validation


train_entries = formatted_entries[:train_size]
val_entries = formatted_entries[train_size:]


# Creating training and validation datasets
train_dataset = Dataset.from_dict({"text": train_entries})
val_dataset = Dataset.from_dict({"text": val_entries})


#from datasets import DatasetDict

#formatted_entries_dataset = Dataset.from_dict({"text": formatted_entries})
#train_size = 1787
#train = formatted_entries_dataset.select(range(train_size)) 
#val = formatted_entries_dataset.select(range(train_size, train_size+300))


#formatted_dataset = DatasetDict({"train": train, "val": val}) 


nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16 # A100
)

#Load Tokenizer
tokenizer= AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', token='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')
# Add Padding Token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load the LLaMA model in 4-bit
model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    token='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',
    quantization_config=nf4_config,
    #use_flash_attention_2=True, 
    device_map="auto",
    trust_remote_code=True,
)


peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
)


# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


args = TrainingArguments(
    output_dir="Llama-2_7B-chat_3_updated",
    overwrite_output_dir=True,
    logging_dir='./results_3_updated',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_checkpointing=False,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=1e-4,
    tf32=True, #A100
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    load_best_model_at_end=True,
    evaluation_strategy='epoch',
    eval_steps=10,
    save_steps=10,
    save_total_limit=30
)


max_seq_length = 2048 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_llama,
    args=args,
)


gc.collect()
torch.cuda.empty_cache()


# train
trainer.train()

# Save Config
trainer.model.config.save_pretrained(trainer.args.output_dir)

# save model
trainer.save_model(trainer.args.output_dir)
