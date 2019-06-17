# Machine Learning Engineer Nanodegree
## Capstone Project

### Summary

In this project, I create a classification model which predicts, from borrower demographic information and company-provided loan conditions and ratings, whether a peer-to-peer loan will be repaid, or not.  To this end, I compared labeled historical Prosper loan data to the repayment status predicted by several supervised learning algorithms, ultimately selecting XGBoost as the algorithm of choice, due to its robust performance in correctly classifying repaid loans as such.

The code can be found in the Jupyter notebook [here](./capstone_project.ipynb).  The written report can be found in the pdf [here](https://eskrav.github.io/udacity-machine-learning/capstone-project/capstone_report.pdf).

### Install

This project requires **Python 3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [xgboost](https://xgboost.readthedocs.io/en/latest/build.html)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

### Run

In a terminal or command window, navigate to the top-level project directory `capstone-project/` (that contains this README) and run one of the following commands:

```bash
ipython notebook capstone_project.ipynb
```  
or
```bash
jupyter notebook capstone_project.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Files

`capstone_project.ipynb`: Python notebook including data processing and analyses

`capstone_report.pdf`: Written report on project

[data exploration](https://github.com/eskrav/udacity-data-analyst/blob/master/explore-and-summarize/explore-and-summarize.Rmd)
