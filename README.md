# Dogs vs Cats 🐶🐱

Model to detect binary categories of similar images. It's designed to train relatively quicky (at least for a neural-network model) without large hardware requirements. Accuracy could likely be improved with additional model complexity. Used on Kaggle's Dog/Kat competition dataset with 80%+ accuracy, rated on Kaggle's external test set as 0.37063 log loss.

This repo can be used for any similar problem if set up with:

- Test data in two labelled folders under 'data/test'
- Train data in two labelled under 'data/train'
- Data you wish to be labelled within 'data/to_predict/all'

First run 'train_weights.py' in order to produce the initial model weighting.
Then run 'model_predictions.py' to produce a table of category predictions/labels.
