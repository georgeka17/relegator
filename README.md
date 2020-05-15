# Relegator Classifier

Machine learning models are used to classify signal and background, a process crucial to many analyses in the physical sciences. Such models are typically trained/optimized by maximizing classification accuracy. Our model seeks to also maximize the statistical significance of the signal sample, which, in the physical sciences, is a major component in determining the merit of an analysis. For data sets in which the characteristics of the signal and background are largely overlapping or for data sets with imbalanced signal/background populations, simultaneously making accurate classification and keeping statistical significance optimally high proves to be difficult with standard approaches. The *relegator classifier* will optimize the statistical significance in signal identification, in which the model has freedom to ignore some regions of input space, and the training loss function combines accuracy and statistical significance.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites
To run the code, you need the following python packages: 'numpy', 'scipy', 'scikit-learn', 'pandas', and tensorflow'. To create plots for the results, you will also need the 'matplotlib' package. The sample code is designed to be run from the command line and this can be done with python's 'sys' module. 

### Installation

This also requires a python environment, which can easily be installed through [Anaconda](https://www.continuum.io/downloads).

## Clone

Clone this repo to your machine using: ''https://github.com/georgeka17/relegator''.


## Features

The sample dataset being used is scikit-learn's *moons* dataset, which makes two half circles that overlap in the following way:

![Alt](/moons_example.png "Moons_Data")

1. run_master_test.py
2. gen_master.py
3. relegator.py

'run_master_test.py' generates a dataset with a set of given parameters using [scikit-learn.datasets.make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html). The dataset is then passed to 'gen_master.py' to prepare it for training.

'gen_master.py' unpacks the information from the dataset that is passed in, prepares it for training, and begins the training process in 'relegator.py'. After the model is finished training, some plots containing the results are generated.

'relegator.py' will set up the model and train the model so that both significance and accuracy are optimized. After training, the results are then brought back to 'gen_master.py' so that they can be visualized.

Users can bring their own dataset, eliminating the need for 'run_master_test.py' and bring their own code that prepares their dataset for training and uses the relegator model.

## Usage
### run_master_test

run_master_test is run through the command line. 

'''bash
python run_master_test.py
'''

### gen_master

### relegator

## Resulting Plots

'gen_master.py' produces some basic plots of the results.

Boundary plot: Here we can see the data being plotted, along with the boundaries that were drawn to classify the regions. The area shaded grey is the *relegated region*, in which these data points were excluded when drawing the boundary between the orange and blue moons. 

![Alt](/noise_0.2_n_train_25000_run3.png "Boundary_Plot")

Histogram:

![Alt](\masses_hist_successful.png "Masses_Histogram")
