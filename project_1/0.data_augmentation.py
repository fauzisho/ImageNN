import os
import cv2
from albumentations import (
    HorizontalFlip, RandomRotate90, ShiftScaleRotate, GaussianBlur, RandomBrightnessContrast, Compose
)
from tqdm import tqdm

# Paths
train_images_path = 'dataset/TrainImages'
augmented_images_path = 'dataset/TrainAugmentedImages'

os.makedirs(f"{augmented_images_path}/pos", exist_ok=True)
os.makedirs(f"{augmented_images_path}/neg", exist_ok=True)

# Define augmentations
def get_augmentations():
    return Compose([
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=5, p=0.5),
        # GaussianBlur(blur_limit=(3, 5), p=0.3),
        # RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ])

# Augment and save images
def augment_and_save(images_path, output_dir, num_augments=5):
    for filename in tqdm(os.listdir(images_path)):
        if filename.endswith('.pgm'):
            img_path = os.path.join(images_path, filename)
            img = cv2.imread(img_path)

            for i in range(num_augments):
                aug = get_augmentations()
                augmented_img = aug(image=img)['image']

                base_name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{base_name}_aug_{i}.png")
                cv2.imwrite(output_path, augmented_img)

augment_and_save(f"{train_images_path}/pos", f"{augmented_images_path}/pos", num_augments=20)
augment_and_save(f"{train_images_path}/neg", f"{augmented_images_path}/neg", num_augments=20)
