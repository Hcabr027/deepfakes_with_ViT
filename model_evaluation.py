from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
import evaluate
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np


# Load the dataset
dataset = load_dataset("JamieWithofs/Deepfake-and-real-images")
test_dataset = dataset['test']  # Use the test split

# Load the processor and model
processor = ViTImageProcessor.from_pretrained('vit-base-deepfake-demo/checkpoint-7004')
model = ViTForImageClassification.from_pretrained('vit-base-deepfake-demo/checkpoint-7004')

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
metric = evaluate.load("accuracy")

# Define the compute_metrics function
def compute_metrics(predictions, references):
   
    return metric.compute(predictions=predictions, references=references)

# Function to predict a single image
def predict(image):
    #image = Image.open(image_path) # Ensure 3 channels
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    return predicted_label, probabilities[0].cpu().tolist()


predictions = []
labels = []


with tqdm(total=len(test_dataset), desc="Evaluating", unit="image") as pbar:
    for idx, example in enumerate(test_dataset):
        image = example['image']
        label = example['label']
        predicted_label, probs = predict(image)
        
        predictions.append(predicted_label)
        labels.append(label)
        
        
        pbar.update(1)


label2id = {"Fake": 0, "Real": 1}  # Predictions are in the format of "Fake" or "Real"

predictions_idx = [label2id[p] for p in predictions]  # Convert predictions to 0/1

# The dataset labels are already integers (0/1)
labels_idx = labels  

# Compute accuracy
metrics = compute_metrics(predictions=np.array(predictions_idx), references=np.array(labels_idx))
accuracy = metrics["accuracy"]

print(f"Test Accuracy: {accuracy * 100:.2f}%")



id2label = {v: k for k, v in label2id.items()}  # Reverse mapping
# Find incorrect predictions
incorrect_indices = [i for i, (pred, true) in enumerate(zip(predictions_idx, labels_idx)) if pred != true]

# Check if there are any incorrect predictions
if incorrect_indices:
    # Get the first incorrect prediction
    first_incorrect_idx = incorrect_indices[0]
    incorrect_example = test_dataset[first_incorrect_idx]
    
    predicted_label = id2label[predictions_idx[first_incorrect_idx]]  
    true_label = id2label[labels_idx[first_incorrect_idx]]            
    # Display the result
    print(f"Incorrect Prediction at index {first_incorrect_idx}:")
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    
    incorrect_image =incorrect_example['image']
    incorrect_image.save("incorrect_image.jpg")
   