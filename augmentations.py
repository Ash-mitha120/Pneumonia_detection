import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(size=(224, 224)),

        A.HorizontalFlip(p=0.5),
        A.CLAHE(p=0.3),
        A.CoarseDropout(max_height=32, max_width=32, p=0.5),

        # ✅ EfficientNet normalization
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(height=224, width=224),

        # ✅ EfficientNet normalization
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
