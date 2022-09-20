import os
import anodet
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

DATASET_PATH = os.path.realpath("/home/wy/anodet/data/")
MODEL_DATA_PATH = os.path.realpath("/home/wy/anodet/distributions/")

paths = [
    os.path.join(DATASET_PATH, "bottle/test/broken_large/000.png"),
    os.path.join(DATASET_PATH, "bottle/test/broken_small/000.png"),
    os.path.join(DATASET_PATH, "bottle/test/contamination/000.png"),
    os.path.join(DATASET_PATH, "bottle/test/good/000.png"),
    os.path.join(DATASET_PATH, "bottle/test/good/001.png"),
]

images = []
for path in paths:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

batch = anodet.to_batch(images, anodet.standard_image_transform, torch.device('cpu'))

embedding_coreset = torch.load(os.path.join(MODEL_DATA_PATH, 'bottle_embedding_coreset_1.pt'))

patch_core = anodet.PatchCore('wide_resnet50', embedding_coreset=embedding_coreset)

image_scores, score_maps = patch_core.predict(batch)

THRESH = 2.3
score_map_classifications = anodet.classification(score_maps, THRESH)
image_classifications = anodet.classification(image_scores, THRESH)
print("Image scores:", image_scores)
print("Image classifications:", image_classifications)

test_images = np.array(images).copy()

boundary_images = anodet.visualization.framed_boundary_images(test_images, score_map_classifications,
                                                              image_classifications, padding=40)
heatmap_images = anodet.visualization.heatmap_images(test_images, score_maps, alpha=0.5)
highlighted_images = anodet.visualization.highlighted_images(images, score_map_classifications, color=(128, 0, 128))

for idx in range(len(images)):
    fig, axs = plt.subplots(1, 4, figsize=(12, 6))
    fig.suptitle('Image: ' + str(idx), y=0.75, fontsize=14)
    axs[0].imshow(images[idx])
    axs[1].imshow(boundary_images[idx])
    axs[2].imshow(heatmap_images[idx])
    axs[3].imshow(highlighted_images[idx])
    plt.savefig(f'./results/result_{idx}.png')
    plt.show()

heatmap_images = anodet.visualization.heatmap_images(test_images, score_maps, alpha=0.5)
tot_img = anodet.visualization.merge_images(heatmap_images, margin=40)
fig, axs = plt.subplots(1, 1, figsize=(10, 6))
plt.imshow(tot_img)
plt.savefig('./results/heatmap.png')
plt.show()