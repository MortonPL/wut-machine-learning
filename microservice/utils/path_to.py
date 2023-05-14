import os
import os.path

def path_to(file, path_, n=1):
    dirname = os.path.dirname(file)
    for _ in range(n):
        dirname, _ = os.path.split(dirname)
    if os.path.basename(dirname) == '.':
        dirname = dirname[:-1]
    return os.path.join(dirname, path_)

def ensure_path(path):
    os.makedirs(path, exist_ok=True)
