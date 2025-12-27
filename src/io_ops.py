from pathlib import Path
import shutil
import os
import stat

def move_to_sorted(image_path, category, sorted_root):
    
    # Move an image into its predicted category folder
    destination_dir = sorted_root / category
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination_path = destination_dir / image_path.name
    shutil.copy2(str(image_path), str(destination_path))

def handle_remove_readonly(func, path):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clear_or_create_folder(target_folder):
    target_folder = Path(target_folder)
    if target_folder.exists():
        shutil.rmtree(target_folder, onexc=handle_remove_readonly)
    target_folder.mkdir(parents=True, exist_ok=True)