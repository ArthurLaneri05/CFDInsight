import subprocess
import platform

""" Opens folder in platform """
def open_folder(path: str) -> None:
    """
    Opens a specified folder.
    """
    system = platform.system()
    
    if system == "Windows":
        subprocess.Popen(f'explorer "{path}"')
    elif system == "Darwin":  # macOS
        subprocess.Popen(['open', path])
    else:  # Linux
        subprocess.Popen(['xdg-open', path])