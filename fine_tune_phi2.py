import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Force CPU execution for MacBook Pro
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables GPU use

# Ensure dataset file exists
dataset_path = "/Users/vijaygowda/Desktop/AI software engineer/huge_cli_dataset.json"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

# Load Phi-2 model and tokenizer
MODEL_NAME = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Fix padding token issue
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA
lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset("json", data_files={"train": dataset_path})["train"]

# ✅ Fix the Tokenization Function
def tokenize_function(examples):
    inputs = [f"User: {inp}\nAssistant: {out}" for inp, out in zip(examples["input"], examples["output"])]
    return tokenizer(inputs, padding="max_length", truncation=True, max_length=512)

# ✅ Fix Dataset Processing with `remove_columns`
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["input", "output"])

# Training Arguments
training_args = TrainingArguments(
    output_dir="./phi2_cli_finetuned",
    evaluation_strategy="no",  # Disables evaluation to prevent error
    save_strategy="epoch",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    fp16=True
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Start fine-tuning
trainer.train()

# Save model and tokenizer
model.save_pretrained("./phi2_cli_finetuned")
tokenizer.save_pretrained("./phi2_cli_finetuned")
