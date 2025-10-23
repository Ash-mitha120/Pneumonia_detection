import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    """
    Advanced augmentation pipeline optimized for medical imaging (X-rays).
    Includes aggressive augmentations for better generalization.
    """
    return A.Compose([
        # Resize and crop
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.8, 1.2), p=1.0),
        
        # Geometric transformations (careful with medical images)
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.15, 
            rotate_limit=10,  # Small rotation for X-rays
            border_mode=0,
            p=0.5
        ),
        
        # Image quality augmentations
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.Equalize(p=1.0),
        ], p=0.5),
        
        # Brightness and contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        # Noise and blur (simulates different X-ray machine qualities)
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        # Image quality degradation
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
        ], p=0.2),
        
        # Pixel-level augmentations
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.5
        ),
        
        # GridDropout for regularization
        A.GridDropout(ratio=0.3, p=0.3),
        
        # Color jittering (mild for grayscale-like X-rays)
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),
        
        # ✅ EfficientNet-B3 normalization (ImageNet stats)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

def get_val_transforms():
    """
    Validation transforms with minimal augmentation.
    Only resize and normalize for consistent evaluation.
    """
    return A.Compose([
        A.Resize(height=224, width=224),
        
        # Optional: Add CLAHE for validation consistency
        # A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        
        # ✅ EfficientNet-B3 normalization (ImageNet stats)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

def get_test_time_augmentation_transforms():
    """
    Test-time augmentation (TTA) for inference.
    Returns multiple versions of the same image for ensemble predictions.
    """
    return A.Compose([
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
