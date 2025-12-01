import os
import shutil
import numpy as np
import pandas as pd
import torch
import evaluate
import PIL
from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import (
    ViTForImageClassification, 
    ViTFeatureExtractor, 
    TrainingArguments, 
    Trainer
)

# Prepare your HailStorm dataset
parent_dir = "/path/to/HailStorm"
classes = os.listdir(parent_dir)
if ".DS_Store" in classes:
    classes.remove(".DS_Store")

meta_df = pd.DataFrame()
for c in classes:
    samples = os.listdir(os.path.join(parent_dir, c))
    samples = [s for s in samples if not s.startswith(".")]
    temp_df = pd.DataFrame(samples, columns=["file_name"])
    temp_df["label"] = c
    meta_df = pd.concat([meta_df, temp_df], ignore_index=True)

new_parent_dir = os.path.join(parent_dir, "data")
os.makedirs(new_parent_dir, exist_ok=True)
meta_df.to_csv(os.path.join(new_parent_dir, "metadata.csv"), index=False)

for c in classes:
    file_names = os.listdir(os.path.join(parent_dir, c))
    for file_name in file_names:
        src = os.path.join(parent_dir, c, file_name)
        dst = os.path.join(new_parent_dir, file_name)
        shutil.move(src, dst)

# Load dataset from the new_parent_dir
dataset = load_dataset(new_parent_dir)
label_names = sorted(set(dataset["train"]["label"]))
dataset = dataset.cast_column("label", ClassLabel(names=label_names))
dataset = dataset.shuffle(seed=42)
split_dataset = dataset["train"].train_test_split(train_size=0.8, seed=42)

hailstorm_ds = DatasetDict({
    "train": split_dataset["train"],
    "test" : split_dataset["test"]
})

# Load your previously trained model from the Hugging Face Hub
MODEL_CKPT = "DunnBC22/vit-base-patch16-224-in21k-weather-images-classification"
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_CKPT)

NUM_OF_EPOCHS = 3
LEARNING_RATE = 2e-4
STEPS = 100
BATCH_SIZE = 16
DEVICE = torch.device("mps")

REPORTS_TO = 'tensorboard'
def transform(example_batch):
    inputs = feature_extractor([img.convert("RGB") for img in example_batch["image"]], 
                               return_tensors="pt")
    inputs["labels"] = example_batch["label"]
    return inputs

prepped_ds = hailstorm_ds.with_transform(transform)

def data_collator(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch])
    }

def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    recall_metric = evaluate.load("recall")
    precision_metric = evaluate.load("precision")
    
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    weighted_f1 = f1_metric.compute(predictions=preds, references=labels, average='weighted')["f1"]
    micro_f1 = f1_metric.compute(predictions=preds, references=labels, average='micro')["f1"]
    macro_f1 = f1_metric.compute(predictions=preds, references=labels, average='macro')["f1"]
    weighted_recall = recall_metric.compute(predictions=preds, references=labels, average='weighted')["recall"]
    micro_recall = recall_metric.compute(predictions=preds, references=labels, average='micro')["recall"]
    macro_recall = recall_metric.compute(predictions=preds, references=labels, average='macro')["recall"]
    weighted_precision = precision_metric.compute(predictions=preds, references=labels, average='weighted')["precision"]
    micro_precision = precision_metric.compute(predictions=preds, references=labels, average='micro')["precision"]
    macro_precision = precision_metric.compute(predictions=preds, references=labels, average='macro')["precision"]
    
    return {
        "accuracy": accuracy,
        "Weighted F1": weighted_f1,
        "Micro F1": micro_f1,
        "Macro F1": macro_f1,
        "Weighted Recall": weighted_recall,
        "Micro Recall": micro_recall,
        "Macro Recall": macro_recall,
        "Weighted Precision": weighted_precision,
        "Micro Precision": micro_precision,
        "Macro Precision": macro_precision,
    }

labels = hailstorm_ds["train"].features["label"].names

model = ViTForImageClassification.from_pretrained(
    MODEL_CKPT,
    num_labels=len(labels),
    id2label={str(i): label for i, label in enumerate(labels)},
    label2id={label: str(i) for i, label in enumerate(labels)}
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

args = TrainingArguments(
    output_dir="vit-hailstorm-finetuned",
    remove_unused_columns=False,
    num_train_epochs=NUM_OF_EPOCHS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    #per_device_eval_batch_size=16,
    learning_rate=LEARNING_RATE,
    report_to=REPORTS_TO,  # or "tensorboard", "wandb", etc.
    load_best_model_at_end=True,
    metric_for_best_model="Weighted F1",
    logging_first_step=True,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=prepped_ds["train"],
    eval_dataset=prepped_ds["test"],
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model("vit-hailstorm-finetuned")
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepped_ds["test"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

print("Finished fine-tuning on the HailStorm dataset!")
