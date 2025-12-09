import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models


model = models.MyModel()
print(model)

ds_train = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale = True)])
)

image, target = ds_train[0]

image = image.unsqueeze(dim = 0)

model.eval()
with torch.no_grad():
    logits = model(image)

probs = logits.softmax(dim = 1)

pred_idx = probs.argmax(dim=1).item()

class_names = ds_train.classes

plt.figure(figsize=(9,4))

plt.subplot(1,2,1)
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"class: {target} {class_names[target]}")
plt.axis('off')

plt.subplot(1,2,2)
plt.bar(range(10), probs[0])
plt.ylim(0,1)
plt.title(f"predicted class: {pred_idx}")
plt.xticks(range(10))

plt.tight_layout()
plt.show()
