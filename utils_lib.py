import io
from torchvision import models
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image

def get_model():
    # Make sure to pass `pretrained` as `True` to use the pretrained weights:
    model = models.densenet121(pretrained=True)
    # Change the last layer to be able to predict 2 classes
    num_ftrs = model.classifier.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model.classifier = nn.Linear(num_ftrs, 2)
    # Load weights
    model.load_state_dict(torch.load('./model/densenet121'))
    # Since we are using our model only for inference, switch to `eval` mode:
    model.eval()
    return model

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)