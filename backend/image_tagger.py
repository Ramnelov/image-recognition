import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip  # Corrected import
from objects import objects

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load and preprocess the image
image_path = "backend/test3.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# List of possible objects

# Encode and compute similarity
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(clip.tokenize(objects).to(device))
#     similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#     values, indices = similarity[0].topk(3)
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(clip.tokenize(objects).to(device))
    similarity = image_features @ text_features.T
    values, indices = similarity[0].topk(3)

# Print top 3 objects
for value, index in zip(values, indices):
    print(f"{objects[index]}: {value.item():.2f}%")
    
    