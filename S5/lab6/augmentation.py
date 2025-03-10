import cv2
import albumentations as A


# Настройка аугментаций
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent={"x": 0.1, "y": 0.1}, rotate=(-45, 45), scale=(0.8, 1.2), p=1.0),
    A.RandomScale(scale_limit=0.1, p=0.5), 
    A.RandomCrop(height=300, width=300, p=0.5), 
    A.RandomBrightnessContrast(p=0.4),
    A.Blur(blur_limit=3, p=0.3),
])

# Количество аугментированных изображений, которое необходимо создать
for index, (image_path, mask_path) in enumerate([
    ('img1.jpg', 'label1.jpg'), 
    ('img2.jpg', 'label2.jpg')
]):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  

    num_augmentations = 50

    for i in range(num_augmentations):
        augmented = transform(image=image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']

        # Сохранение аугментированных изображений и масок
        cv2.imwrite(f'images/image{index}_{i}.jpg', augmented_image)
        cv2.imwrite(f'labels/image{index}_{i}.jpg', augmented_mask)


print(f"Done")
