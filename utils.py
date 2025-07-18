from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os

dog_list = {
    "푸들": "static/dog_images/푸들.jpg",
    "시바견": "static/dog_images/시바견.jpg",
    "리트리버": "static/dog_images/리트리버.jpg"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_most_similar_dog(user_image_file):
    user_image = Image.open(user_image_file).convert("RGB")
    dog_images = [Image.open(path).convert("RGB") for path in dog_list.values()]
    all_images = [user_image] + dog_images

    inputs = processor(images=all_images, return_tensors="pt", padding=True).to(device)
    image_features = model.get_image_features(**inputs)

    # [0] = user, [1:] = dogs
    user_vec = image_features[0].unsqueeze(0)         # shape: (1, D)
    dog_vecs = image_features[1:]                     # shape: (N, D)

    similarities = torch.nn.functional.cosine_similarity(user_vec, dog_vecs)
    max_idx = torch.argmax(similarities).item()
    print("similarities:", similarities.tolist())
    return list(dog_list.keys())[max_idx]
