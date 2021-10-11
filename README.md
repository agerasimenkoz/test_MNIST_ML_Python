# MNIST with CNN Keras
Console application for training model CNN and predictions for your pictures of handwritten numbers
## How to use
* Install requirements.txt `pip -r requirements.txt`
* Downloads dataset MNIST to temp project path `bash downloads_mnist.sh`
* Train CNN model `python mnist.py train --dataset tmp/mnist_png/training --model tmp/model/model.h5`
* Evaluate CNN model `python mnist.py test --dataset tmp/mnist_png/testing --model tmp/model/model.h5`
* Predict for your pictures `python mnist.py predict --model tmp/model/model.h5  --input path/to/dataset.csv --output path/to/output/predictions.csv` 


## Source code
* [mnist.py](mnist.py) contains console logic
* [downloads_mnist.sh](downloads_mnist.sh) downloads MNIST dataset
