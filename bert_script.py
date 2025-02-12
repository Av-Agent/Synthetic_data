import torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, BertForSequenceClassification, DataCollatorWithPadding
import numpy as np
import matplotlib.pyplot as plt
import evaluate
import pandas as pd
import scipy
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding

# Define input and output file names
input_csv = "numeric_label.csv"
train_csv = "train.csv"

# Read the CSV file
df = pd.read_csv(input_csv)
df.columns = ["id", "query", "label"]
df = df.drop(columns=["id"])
df.to_csv(train_csv, index=False)

#print(f"Dataset saved as {train_csv} ({len(df)} samples)")


# Load the CSV files into pandas DataFrames
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('train.csv')
test_df = pd.read_csv('train.csv')

# make dataset for HuggingFace
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)


# train dataset
# # Model parameters start 
learning_rate = 4.00e-05
warmup_proportion = 0.1
train_batch_size = 16
eval_batch_size = 16
num_train_epochs = 5
gradient_accumulation_steps = 1
huggingface_modelname = "bert-base-uncased"

# define tokeniser

tokenizer = AutoTokenizer.from_pretrained(huggingface_modelname)
def tokenize_function(examples):
    return tokenizer(examples["query"],truncation=True)
#tokenize datasets 
train_datasets = train_dataset.map(tokenize_function,batched = True)
val_datasets = val_dataset.map(tokenize_function,batched = True)
test_datasets = test_dataset.map(tokenize_function,batched = True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

#print("Unique labels in dataset:", train_df["label"].unique())
#print("Number of unique labels:", len(train_df["label"].unique()))



model = BertForSequenceClassification.from_pretrained(huggingface_modelname,num_labels=len(train_df["label"].unique()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_args = TrainingArguments(
    output_dir="test_trainer", #model checkpoints, logs, and other training outputs will be saved. ex-pytorch_model.bin
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    weight_decay=0.01,  #L2 regularization coefficient applied to prevent overfitting by penalizing large weights.
    evaluation_strategy="epoch", #Defines how often evaluation is performed
    save_strategy="epoch", #pecifies how often model checkpoints are saved
    gradient_accumulation_steps=gradient_accumulation_steps,
    load_best_model_at_end=True,
    )

accuracy_metric  = evaluate.load("accuracy")
#f1_metric = evaluate.load("f1")
#The structure of eval_pred is usually a tuple: (predictions, labels).
#eval_pred argument is supplied by the Trainer and contains 1- Predictions: Model output predictions (usually logits or probability scores).2- Labels: Ground truth labels from the dataset.
def compute_metrics(eval_pred): 
    logits, labels = eval_pred  #(predictions, labels)

    predictions = np.argmax(logits, axis=-1) #Converts raw model outputs (logits) into class predictions, selects the highest logit value
        # Calculate accuracy
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    
    # Calculate F1 score (macro or weighted based on your use case)
    #f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    # Combine metrics into a single dictionary
    metrics = {
        "accuracy": accuracy["accuracy"],
        #"f1": f1["f1"],
    }
    return metrics

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_datasets.shuffle(seed=42),
    eval_dataset=val_datasets.shuffle(seed=42),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

pred_train = trainer.predict(train_datasets)
pred_val = trainer.predict(val_datasets)
pred_test = trainer.predict(test_datasets)

# Load metrics
accuracy_metric = evaluate.load("accuracy")
#f1_metric = evaluate.load("f1")

# Extract the predicted labels from the logits
pred_train_labels = np.argmax(pred_train.predictions, axis=-1)
pred_val_labels = np.argmax(pred_val.predictions, axis=-1)
pred_test_labels = np.argmax(pred_test.predictions, axis=-1)

# Calculate accuracy and F1 score on train dataset
train_accuracy = accuracy_metric.compute(predictions=pred_train_labels, references=pred_train.label_ids)
#train_f1 = f1_metric.compute(predictions=pred_train_labels, references=pred_train.label_ids, average="weighted")

# Calculate accuracy and F1 score on validation dataset
val_accuracy = accuracy_metric.compute(predictions=pred_val_labels, references=pred_val.label_ids)
#val_f1 = f1_metric.compute(predictions=pred_val_labels, references=pred_val.label_ids, average="weighted")

# Calculate accuracy and F1 score on test dataset
test_accuracy = accuracy_metric.compute(predictions=pred_test_labels, references=pred_test.label_ids)
#test_f1 = f1_metric.compute(predictions=pred_test_labels, references=pred_test.label_ids, average="weighted")

# Print the results
print(f"Train Accuracy: {train_accuracy['accuracy']}")
print(f"Validation Accuracy: {val_accuracy['accuracy']}")
print(f"Test Accuracy: {test_accuracy['accuracy']}")


output_dir = "./saved_model_llama"

# Save the model
trainer.save_model(output_dir)

# Save the tokenizer (optional, but useful for reloading later)
tokenizer.save_pretrained(output_dir)

#print(f"Model and tokenizer saved to {output_dir}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Specify the path to your saved model
saved_model_path = "./saved_model_llama"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)

# Define a function for prediction
def predict_class(user_input):
    # Tokenize the user input
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    return predicted_class

if __name__ == "__main__":
    # Run inference in a loop to allow user input
    print("Type 'exit' to quit the program.")
    while True:
        user_input = input("Enter text for classification: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        predicted_class = predict_class(user_input)
        #print(f"Predicted class: {predicted_class}")
        print(f"User Input: '{user_input}' -> Predicted Class: {predicted_class}")



