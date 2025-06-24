from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Load SST2 sentiment dataset
dataset = load_dataset("glue", "sst2")

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize sentences
def preprocess_function(batch):
    tokenized = tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)
    tokenized["label"] = batch["label"]
    return tokenized


encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training config
training_args = TrainingArguments(
    output_dir="./trained_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=16,
    num_train_epochs=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=encoded_dataset["validation"]
)

# Train and save
trainer.train()
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
