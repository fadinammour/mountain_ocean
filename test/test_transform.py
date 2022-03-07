import sys

path_transform = '../'
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, path_transform)

from utils_lib import transform_image

with open("../_static/img/sample_file.jpg", 'rb') as f:
    image_bytes = f.read()
    tensor = transform_image(image_bytes=image_bytes)
    print(tensor)