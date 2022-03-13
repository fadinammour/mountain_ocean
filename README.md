# Mountain v/s Ocean

Deep Learning Web App to classify Ocean and Mountain Images using PyTorch, Flask and Heroku.

## Context

This project has been developped as a technical test during the recruitment process at Ideta in March 2022.

## Installation and Use

Locally :

1. (Optional) Create a virtual environment.
2. Clone the git repository.
3. In an open terminal, activate the virtual environment, change directory to the cloned repository and run `pip -r requirements.txt`.
4. Run `FLASK_ENV=development FLASK_APP=app.py flask run`.

Remotely : Click on the badge [![Mountain v/s Ocean](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Heroku_logo.svg/2560px-Heroku_logo.svg.png =100)](https://mountain-vs-ocean.herokuapp.com/)

## Approach

To tackle this project, I divided the work into 3 main parts : Web Design, Web App and Deep Learning

### Web Design & App

I started by choosing Flask to implement the Web Design & App parts, for that purpose, I followed this tutorial :

https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html

Then I was lucky to find that this tutorial was extended to Heroku which helped hosting the Web App :

https://github.com/avinassh/pytorch-flask-api-heroku

I decided to keep the design minimalistic. The main improvements that can be brought for the webpage are :

* adding examples to be able to easily test the application.
* adding more documentation and context to the page.
* adding more options to upload an image (different formats, URL, ...)

### Deep Learning

To build the Deep Learning classifier, I mainly got inspired by the following GitHub repository : https://github.com/dinhanhthi/mountain-vs-beach.

#### Dataset

For the dataset, I carefully analyzed all the default datasets avaible in PyTorch (https://pytorch.org/vision/stable/datasets.html). The only one that was relevant to the project was Places365. And in a similar fashion to Anh-Thi DINH's approach, I selected all the categories related to the Ocean (Beach, Beach House, Ocean) and all the categories related to Mountain (Mountain, Mountain path, Mountain snowy). This gave me a perfectly balanced dataset of 30,000 images. I used the small version of the dataset which contains images of standardized size of 256x256.

In the file `utils_lib.py`, I coded a new class that automatically generates a suited dataset for our project.

#### Network Architecture and Learning Approach

Since the dataset size is relatively small, I opted for a Transfer Learning approach. I decided to simply implement the approach given in the PyTorch example : https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html.

The network architecture that was used is Densenet121. However, it would be interesting to investigate different architectures.

For the training, I used the following parameters :

* Number of epoch : 25
* Batch size : 128
* Optimizer : Stochastic Gradient Descent (SGD)

It would be also interesting to try different values for these parameter like 32, 64 and 256 for the batch size and Adam for the optimizer.

The accuracy of the network on the validation dataset is : 95.33%.
And the training was performed on my MacBook Pro 2021 and took around : ~9 hours.

It would be interesting to optimize the training by taking advantage of multi-processing for example.

## What I learned

By performing this project, I learning many valuable skills :

* Coding a Web App with Flask,
* Designing a basic Web Page with HTML,
* Running a Website on Heroku,
* Transfer Learning with PyTorch.
