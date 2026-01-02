from pathlib import Path
import shutil
import os
import stat

def handle_remove_readonly(func, path):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clear_or_create_folder(target_folder):
    target_folder = Path(target_folder)
    if target_folder.exists():
        shutil.rmtree(target_folder, onerror=handle_remove_readonly)
    target_folder.mkdir(parents=True, exist_ok=True)