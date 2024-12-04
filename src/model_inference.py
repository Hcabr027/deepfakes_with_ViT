from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

image_path = '/home/hcabrera/Projects/image-2.jpg'
image = Image.open(image_path)

processor = ViTImageProcessor.from_pretrained('vit-base-deepfake-demo/checkpoint-7004')
model = ViTForImageClassification.from_pretrained('vit-base-deepfake-demo/checkpoint-7004')


inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

probabilities = torch.nn.functional.softmax(logits, dim=1)


predicted_class_idx = torch.argmax(probabilities, dim=1).item()


predicted_label = model.config.id2label[predicted_class_idx]

print(f"Predicted Label: {predicted_label}")
print(f"Probabilities: {probabilities}")
