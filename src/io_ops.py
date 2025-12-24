from pathlib import Path
import shutil

def move_to_sorted(image_path, category, sorted_root):
    
    # Move an image into its predicted category folder
    destination_dir = sorted_root / category
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination_path = destination_dir / image_path.name
    shutil.copy2(str(image_path), str(destination_path))