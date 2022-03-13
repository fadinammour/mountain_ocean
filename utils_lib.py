import io
from torchvision import models
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.datasets import Places365
from typing import Optional, Callable
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

class Moucean(Places365):
    """
    A unsupervised anomaly detection version of MNIST.
    In train mode, only normal digit are available.
    In test mode, all digits are available
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        split: Optional[Callable] = 'train-standard',
        small: bool = True,
        download: bool = False,
    ):
        """

        :param train: train/test set
        :param anomaly_categories: digit categories to exclude from train set
        """
        super().__init__(root=root, transform=transform,
                         target_transform=target_transform,
                         split=split, small=small, download=download)
        self.ocean_categories = [243,48,49]
        self.mountain_categories = [232,233,234]
        # generate "ocean","mountain" categories
        self.targets = []
        self.ocean_ind = []
        self.mountain_ind = []
        self.moucean_ind = []
        imgs =[]
        
        for ind,im in enumerate(self.imgs):
            _,cat = im
            if cat in self.ocean_categories:
                self.ocean_ind += [ind]
                self.moucean_ind += [ind]
                imgs += [(self.imgs[ind][0],1)]
                self.targets += [1]
            elif cat in self.mountain_categories:
                self.mountain_ind += [ind]
                self.moucean_ind += [ind]
                imgs += [(self.imgs[ind][0],0)]
                self.targets += [0]
        # remove other categories
        self.imgs = imgs
        self.class_to_idx = {'Mountain':0,'Ocean':1}
        self.classes = ['Mountain','Ocean']