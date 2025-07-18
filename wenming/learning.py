import transformers
import torch

model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
from PIL import Image

images = [Image.open("/home/wenming/projects/transformers/wenming/image.png")]
possible_classes = ["an image of a bird", "an image of a dog", "an image of a cat"]

descriptions = [f"a photo of {c}" for c in possible_classes]
inputs = processor(
    text=descriptions,
    images=images,
    return_tensors="pt",
    padding=True,
    truncation=True,
)
outputs = model(**inputs)
print(outputs)
