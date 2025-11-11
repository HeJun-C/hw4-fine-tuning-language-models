import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
# import evaluate
import random
import argparse
from utils import *
import os
from torch.cuda.amp import autocast, GradScaler

# Set seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Tokenize the input
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Core training function
def do_train(args, model, train_dataloader, save_dir="./out"):
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    # Add gradient scaler for mixed precision training
    scaler = GradScaler()
    
    model.train()
    progress_bar = tqdm(range(num_training_steps))

    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Standard PyTorch training loop with zero_grad -> forward -> backward -> step -> scheduler
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # Move batch to device (non_blocking for better performance)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Zero accumulated gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision (labels already in batch as "labels")
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss

            # Backprop with gradient scaling
            scaler.scale(loss).backward()

            # Optimizer step with gradient scaling
            scaler.step(optimizer)
            scaler.update()
            
            # LR scheduler step
            lr_scheduler.step()

            # Update progress bar
            progress_bar.update(1)

    ##### YOUR CODE ENDS HERE ######

    print("Training completed...")
    print("Saving Model....")
    model.save_pretrained(save_dir)

    return


# Core evaluation function
# Core evaluation function
def do_eval(eval_dataloader, output_dir, out_file):
    from sklearn.metrics import accuracy_score
    
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    
    out_file_handle = open(out_file, "w")

    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.no_grad():
            with autocast():
                outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

        # write to output file
        for pred, label in zip(predictions, batch["labels"]):
            out_file_handle.write(f"{pred.item()}\n")
            out_file_handle.write(f"{label.item()}\n")
    
    out_file_handle.close()
    
    accuracy = accuracy_score(all_labels, all_predictions)
    score = {'accuracy': accuracy}

    return score


# Create a dataloader for the augmented training dataset
def create_augmented_dataloader(args, dataset):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Take the original train split and add 5k randomly transformed examples
    # 1) Transform the full train split
    transformed_train = dataset["train"].map(custom_transform, batched=True, load_from_cache_file=False, num_proc=4)

    # 2) Sample 5k transformed examples
    transformed_subset = transformed_train.shuffle(seed=42).select(range(5000))

    # 3) Concatenate original + transformed_subset (raw text/labels)
    combined_raw = datasets.concatenate_datasets([dataset["train"], transformed_subset])

    # 4) Tokenize and format for the model
    tokenized = combined_raw.map(tokenize_function, batched=True, load_from_cache_file=False, num_proc=4)
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    # 5) Build dataloader with optimized settings
    train_dataloader = DataLoader(
        tokenized, 
        shuffle=True, 
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )

    ##### YOUR CODE ENDS HERE ######

    return train_dataloader


# Create a dataloader for the transformed test set
def create_transformed_dataloader(args, dataset, debug_transformation):
    # Print 5 random transformed examples
    if debug_transformation:
        small_dataset = dataset["test"].shuffle(seed=42).select(range(5))
        small_transformed_dataset = small_dataset.map(custom_transform, load_from_cache_file=False)
        for k in range(5):
            print("Original Example ", str(k))
            print(small_dataset[k])
            print("\n")
            print("Transformed Example ", str(k))
            print(small_transformed_dataset[k])
            print('=' * 30)

        exit()

    transformed_dataset = dataset["test"].map(custom_transform, batched=True, load_from_cache_file=False, num_proc=4)
    transformed_tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True, load_from_cache_file=False, num_proc=4)
    transformed_tokenized_dataset = transformed_tokenized_dataset.remove_columns(["text"])
    transformed_tokenized_dataset = transformed_tokenized_dataset.rename_column("label", "labels")
    transformed_tokenized_dataset.set_format("torch")

    transformed_val_dataset = transformed_tokenized_dataset
    eval_dataloader = DataLoader(
        transformed_val_dataset, 
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )

    return eval_dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--train_augmented", action="store_true", help="train a model on the augmented training data")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--eval_transformed", action="store_true", help="evaluate model on the transformed test set")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--debug_train", action="store_true",
                        help="use a subset for training to debug your training loop")
    parser.add_argument("--debug_transformation", action="store_true",
                        help="print a few transformed examples for debugging")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)  # Increased from 8 to 32 for better GPU utilization

    args = parser.parse_args()

    global device
    global tokenizer

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Tokenize the dataset with parallel processing
    dataset = load_dataset("imdb")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)

    # Prepare dataset for use by model
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(4000))
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    # Create dataloaders for iterating over the dataset with optimized settings
    if args.debug_train:
        train_dataloader = DataLoader(
            small_train_dataset, 
            shuffle=True, 
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True
        )
        eval_dataloader = DataLoader(
            small_eval_dataset, 
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True
        )
        print(f"Debug training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")
    else:
        train_dataloader = DataLoader(
            tokenized_dataset["train"], 
            shuffle=True, 
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True
        )
        eval_dataloader = DataLoader(
            tokenized_dataset["test"], 
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True
        )
        print(f"Actual training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")

    # Train model on the original training dataset
    if args.train:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out")
        # Change eval dir
        args.model_dir = "./out"

    # Train model on the augmented training dataset
    if args.train_augmented:
        train_dataloader = create_augmented_dataloader(args, dataset)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)
        do_train(args, model, train_dataloader, save_dir="./out_augmented")
        # Change eval dir
        args.model_dir = "./out_augmented"

    # Evaluate the trained model on the original test dataset
    if args.eval:
        base = os.path.basename(os.path.normpath(args.model_dir))
        out_file = f"{base}.original.txt"   # matches assignment naming
        score = do_eval(eval_dataloader, args.model_dir, out_file)
        print("Score: ", score)

    # Evaluate the trained model on the transformed test dataset
    if args.eval_transformed:
        base = os.path.basename(os.path.normpath(args.model_dir))
        out_file = f"{base}.transformed.txt"   # consistent naming
        eval_transformed_dataloader = create_transformed_dataloader(args, dataset, args.debug_transformation)
        score = do_eval(eval_transformed_dataloader, args.model_dir, out_file)
        print("Score: ", score)