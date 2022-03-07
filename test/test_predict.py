import sys

path_predict = '../app/'
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, path_predict)

from main import get_prediction

with open("../_static/img/sample_file.jpg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))