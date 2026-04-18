import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

    # 1. Load the Enron Email Spam Dataset
    # We use a small subset (1000 train, 500 test) to make training fast for your assignment
dataset = load_dataset("SetFit/enron_spam")
train_ds = dataset["train"].shuffle(seed=42).select(range(1000)) 
test_ds = dataset["test"].shuffle(seed=42).select(range(500))

    # 2. Tokenize the text
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_ds.map(tokenize_function, batched=True)
tokenized_test = test_ds.map(tokenize_function, batched=True)

    # 3. Load the Base Model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # 4. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1, # Keep it at 1 for quick assignment testing
    weight_decay=0.01,
    )

    # 5. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    )

    # 6. Train and Save
print("Starting Training...")
trainer.train()

print("Evaluating Model...")
print(trainer.evaluate())

print("Saving the fine-tuned model...")
model.save_pretrained("./spam_model")
tokenizer.save_pretrained("./spam_model")
print("Model saved to ./spam_model. Ready for Flask!")
    