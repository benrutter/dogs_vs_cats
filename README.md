# Dogs vs Cats 🐶🐱

Models to detect binary categories of similar images. It's designed to train relatively quicky (at least for a neural-network model) without large hardware requirements. Accuracy could likely be improved with additional model complexity.

There are two models here, a pre-trained resnet, and a convolutional network from scratch.

The from-scratch network has been used on Kaggle's Dog/Kat competition dataset with 80%+ accuracy, rated on Kaggle's external test set as 0.37063 log loss.
The pretrained resnet performs on the same dataset with 95%+ accuracy, rated on Kaggle's external test set as 0.11149 log loss.

This repo can be used for any similar problem if set up with:

- Test data in two labelled folders under 'data/test'
- Train data in two labelled under 'data/train'
- Data you wish to be labelled within 'data/to_predict/all'
- Example / Model output (with all labels) in 'sample_submission.csv'

First run 'cnn.py'/'resnet.py' in order to produce the initial model weighting.
Then run 'cnn_predictions.py'/'resnet_predictions.py' to produce a table of category predictions/labels.
