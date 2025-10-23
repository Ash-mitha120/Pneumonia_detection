"""
Setup Verification Script for Pneumonia Detection Project
Run this before training to ensure everything is configured correctly.
"""

import sys
import os

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def print_status(check, status, message):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {check}: {message}")

def verify_python_version():
    print_header("Python Version Check")
    version = sys.version_info
    status = version.major == 3 and version.minor >= 8
    print_status("Python Version", status, f"{version.major}.{version.minor}.{version.micro}")
    if not status:
        print("   ⚠️  Python 3.8+ required. Current version may have compatibility issues.")
    return status

def verify_packages():
    print_header("Package Installation Check")
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'efficientnet_pytorch': 'EfficientNet',
        'albumentations': 'Albumentations',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tensorboard': 'TensorBoard',
        'optuna': 'Optuna',
        'tqdm': 'TQDM'
    }
    
    all_installed = True
    for module, name in packages.items():
        try:
            __import__(module)
            print_status(name, True, "Installed")
        except ImportError:
            print_status(name, False, "NOT FOUND")
            print(f"   Install with: pip install {name.lower()}")
            all_installed = False
    
    return all_installed

def verify_cuda():
    print_header("CUDA & GPU Check")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print_status("CUDA Available", cuda_available, 
                    "GPU acceleration enabled" if cuda_available else "Using CPU (slower)")
        
        if cuda_available:
            print(f"   🔹 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   🔹 CUDA Version: {torch.version.cuda}")
            print(f"   🔹 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("   ⚠️  Training on CPU will be significantly slower.")
            print("   💡 Consider using Google Colab or Kaggle for free GPU access.")
        
        return True
    except Exception as e:
        print_status("PyTorch", False, str(e))
        return False

def verify_dataset():
    print_header("Dataset Structure Check")
    
    # Update these paths based on train_model.py
    base_paths = [
        r"C:\Users\ASUS\Downloads\archive\chest_xray",
        r"D:\pneumonia detection\chest_xray",
        r"chest_xray"
    ]
    
    found = False
    for base_path in base_paths:
        if os.path.exists(base_path):
            found = True
            print_status("Dataset Directory", True, base_path)
            
            # Check subdirectories
            required_dirs = ['train', 'val', 'test']
            for dir_name in required_dirs:
                dir_path = os.path.join(base_path, dir_name)
                if os.path.exists(dir_path):
                    # Check class folders
                    normal_path = os.path.join(dir_path, 'NORMAL')
                    pneumonia_path = os.path.join(dir_path, 'PNEUMONIA')
                    
                    if os.path.exists(normal_path) and os.path.exists(pneumonia_path):
                        normal_count = len([f for f in os.listdir(normal_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                        pneumonia_count = len([f for f in os.listdir(pneumonia_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                        print_status(f"{dir_name.upper()} set", True, 
                                   f"NORMAL: {normal_count}, PNEUMONIA: {pneumonia_count}")
                    else:
                        print_status(f"{dir_name.upper()} set", False, "Class folders missing")
                else:
                    print_status(f"{dir_name.upper()} set", False, "Directory not found")
            
            break
    
    if not found:
        print_status("Dataset Directory", False, "Not found in expected locations")
        print("\n   📁 Expected structure:")
        print("   chest_xray/")
        print("   ├── train/")
        print("   │   ├── NORMAL/")
        print("   │   └── PNEUMONIA/")
        print("   ├── val/")
        print("   │   ├── NORMAL/")
        print("   │   └── PNEUMONIA/")
        print("   └── test/")
        print("       ├── NORMAL/")
        print("       └── PNEUMONIA/")
        print("\n   ⚠️  Please update paths in train_model.py")
    
    return found

def verify_project_files():
    print_header("Project Files Check")
    
    required_files = {
        'train_model.py': 'Training script',
        'augmentations.py': 'Data augmentation',
        'model_utils.py': 'Dataset utilities',
        'cutmix_utils.py': 'CutMix augmentation',
        'evaluate_model.py': 'Model evaluation',
        'requirements.txt': 'Dependencies list'
    }
    
    all_present = True
    for file, description in required_files.items():
        exists = os.path.exists(file)
        print_status(file, exists, description)
        if not exists:
            all_present = False
    
    return all_present

def test_imports():
    print_header("Module Import Test")
    
    try:
        from augmentations import get_train_transforms, get_val_transforms
        print_status("augmentations.py", True, "Import successful")
        
        from model_utils import CustomImageDataset
        print_status("model_utils.py", True, "Import successful")
        
        from cutmix_utils import cutmix_data, cutmix_criterion
        print_status("cutmix_utils.py", True, "Import successful")
        
        return True
    except Exception as e:
        print_status("Module Imports", False, str(e))
        return False

def estimate_training_time():
    print_header("Training Time Estimation")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            
            if 'rtx 3090' in gpu_name or 'a100' in gpu_name:
                time = "3-4 hours"
            elif 'rtx 3080' in gpu_name or 'rtx 3070' in gpu_name:
                time = "4-6 hours"
            elif 'rtx 2080' in gpu_name or 'rtx 2070' in gpu_name:
                time = "6-8 hours"
            elif 'gtx 1080' in gpu_name or 'gtx 1070' in gpu_name:
                time = "8-12 hours"
            else:
                time = "4-10 hours (varies by GPU)"
            
            print(f"   ⏱️  Estimated training time: {time}")
            print(f"   📊 For ~15,000 training images, 50 epochs, batch size 32")
        else:
            print("   ⏱️  CPU training: 2-5 days (NOT RECOMMENDED)")
            print("   💡 Strongly recommend using GPU")
    except:
        pass

def main():
    print("\n" + "="*60)
    print("  🔍 PNEUMONIA DETECTION - SETUP VERIFICATION")
    print("="*60)
    
    checks = []
    
    # Run all verification checks
    checks.append(("Python", verify_python_version()))
    checks.append(("Packages", verify_packages()))
    checks.append(("CUDA/GPU", verify_cuda()))
    checks.append(("Dataset", verify_dataset()))
    checks.append(("Project Files", verify_project_files()))
    checks.append(("Module Imports", test_imports()))
    
    # Summary
    print_header("Verification Summary")
    
    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    
    for check_name, status in checks:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {check_name}")
    
    print(f"\n   📊 Passed: {passed}/{total} checks")
    
    if passed == total:
        print("\n   🎉 All checks passed! You're ready to train!")
        print("\n   🚀 Start training with: python train_model.py")
        print("   📊 Monitor with: tensorboard --logdir=runs")
        estimate_training_time()
    else:
        print("\n   ⚠️  Some checks failed. Please resolve issues before training.")
        print("   📖 Check TRAINING_GUIDE.md for detailed instructions.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
