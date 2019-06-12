# Machine Learning Engineer Nanodegree
# Supervised Learning
## Project: Finding Donors for CharityML

## Summary

In this project, I explore, clean, and transform census data on a set of potential charity donors.  I then use several supervised learning models (Random Forest, Logistic Regression, AdaBoost) to predict individual income class (and therefore, likelihood of making a donation, in this use case).  I compare model performance directly using default settings for each algorithm, and finally perform grid search to optimize the best-performing model (AdaBoost, in this case).  I then determine how well the optimal model will train if only the 5 most predictive features are used.

The Jupyter notebook for this project can be found [here](./finding_donors.ipynb), while the html export of this notebook can be found [here](https://eskrav.github.io/udacity-machine-learning/finding-donors/finding_donors.html).

## Software Requirements

This project uses the following software and Python libraries:

- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer.
