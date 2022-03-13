from flask import Flask, render_template, jsonify, request
import json
from utils_lib import transform_image, get_model

app = Flask(__name__)

model = get_model()

imagenet_class_index = json.load(open('./static/imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    temp, y_hat = outputs.max(1)
    if temp < 1 :
        return 'Undetermined'
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        return render_template('result.html', class_name=class_name)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()