from config.cgiar import config
from loader.input_loader import cgiar_paths_labels
import os
import random
from sklearn.preprocessing import LabelEncoder

random.seed(config.SEED)


paths, labels = cgiar_paths_labels(config.CGIAR_DATASET, os.path.join(config.CGIAR_METADATA, 'train.csv'))

# binary classification approach
drought_images_path = []
non_drought_images_path = []
drought_labels = []
non_drought_labels = []

for path, label in zip(paths, labels ):
    if label == 'DR':
        drought_images_path.append(path)
        drought_labels.append(label)
    else:
        non_drought_images_path.append(path)
        non_drought_labels.append(label)


print(drought_images_path[:5])
print(drought_labels[:5])
print()
print(non_drought_images_path[:5])
print(non_drought_labels[:5])
print()
print(len(drought_images_path))
print(len(non_drought_images_path))


for i, l in enumerate(non_drought_labels):
    non_drought_labels[i] = 'B'

print()
print(len(non_drought_labels))
print(non_drought_labels[:5])

image_paths = drought_images_path + non_drought_images_path
labels = drought_labels + non_drought_labels

print()
print(len(image_paths))
print(len(labels))

print(set(labels))


# shuffling
data = list(zip(image_paths, labels))
random.shuffle(data)

images_path, labels = zip(*data)

images_path = list(images_path)
labels = list(labels)

print(images_path[:30])
print(labels[:30])

# Encode labels
le = LabelEncoder()
labels = le.fit_transform(labels)
label_names = le.classes_

print(labels[:30])
print(label_names)









