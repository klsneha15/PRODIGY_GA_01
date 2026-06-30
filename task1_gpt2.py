from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset

print("Loading GPT-2 model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model     = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded!")

train_text = """
Artificial intelligence is transforming the world.
Machine learning models learn from data automatically.
Deep learning uses neural networks with many layers.
Natural language processing helps computers understand text.
Generative AI can create new content like text and images.
Python is the most popular language for AI development.
PyTorch and TensorFlow are the top deep learning frameworks.
Transfer learning allows models to be reused for new tasks.
Fine-tuning adapts a pretrained model to a specific domain.
Large language models are trained on billions of text tokens.
"""

lines = [line.strip() for line in train_text.strip().split("\n") if line.strip()]
print(f"Training lines: {len(lines)}")

dataset = Dataset.from_dict({"text": lines})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print(f"Dataset size: {len(tokenized_dataset)} samples")

training_args = TrainingArguments(
    output_dir="output",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=100,
    save_total_limit=1,
    logging_steps=10,
    use_cpu=True
)

print("\nStarting fine-tuning...")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset
)
trainer.train()
print("Training complete!")

print("\nGenerating text...")
inputs = tokenizer.encode("Artificial intelligence", return_tensors="pt")
outputs = model.generate(
    inputs,
    max_length=80,
    num_return_sequences=3,
    temperature=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

print("\n--- Generated Text ---")
for i, output in enumerate(outputs):
    print(f"\nSample {i+1}:")
    print(tokenizer.decode(output, skip_special_tokens=True))