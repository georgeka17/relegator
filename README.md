# Relegator Classifier

Machine learning models are used to classify signal and background, a process crucial to many analyses in the physical sciences. Such models are typically trained/optimized by maximizing classification accuracy. Our model seeks to also maximize the statistical significance of the signal sample, which, in the physical sciences, is a major component in determining the merit of an analysis. For data sets in which the characteristics of the signal and background are largely overlapping or for data sets with imbalanced signal/background populations, simultaneously making accurate classification and keeping statistical significance optimally high proves to be difficult with standard approaches. The *relegator classifier* will optimize the statistical significance in signal identification, in which the model has freedom to ignore some regions of input space, and the training loss function combines accuracy and statistical significance.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites
To run the code, you need the following python packages: `numpy`, `scipy`, `scikit-learn`, `pandas`, and `tensorflow`. To create plots for the results, you will also need the `matplotlib` package. For this use case, you will also need the modules from [moons_tools_2.py](https://github.com/georgeka17/relegator/blob/master/moons_tools_2.py) for plotting functionality and [make_datasets_2_gen.py](https://github.com/georgeka17/relegator/blob/master/make_datasets_2_gen.py) to generate a new dataset. The sample code is designed to be run from the command line and this can be done with python's `sys` module. 

It is important to use TensorFlow 2.0 so that we can utilize the *eager execution* feature, which allows us to monitor the results in between training and makes the neural network more efficient, which reduces our train time. 

### Installation

This also requires a python environment, which can easily be installed through [Anaconda](https://www.continuum.io/downloads).

## Clone

Clone this repo to your machine using: https://github.com/georgeka17/relegator.

## Features

The sample dataset being used is scikit-learn's *moons* dataset, which makes two half circles that overlap in the following way:

![Alt](/moons_example.png "Moons_Data")

1. run_master_test.py
2. gen_master.py
3. relegator.py

`run_master_test.py` generates a dataset with a set of given parameters using [scikit-learn.datasets.make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html). The dataset is then passed to `gen_master.py` to prepare it for training.

`gen_master.py` unpacks the information from the dataset that is passed in, prepares it for training, and begins the training process in `relegator.py`. After the model is finished training, some plots containing the results are generated.

`relegator.py` will set up the model and train the model so that both significance and accuracy are optimized. After training, the results are then brought back to `gen_master.py` so that they can be visualized.

Users can bring their own dataset, eliminating the need for `run_master_test.py` and bring their own code that prepares their dataset for training and uses the relegator model.

## Usage
### run_master_test

`run_master_test` is run through the command line. 

```
python run_master_test.py
```
The parameters specified in the json string can be altered. Some notable features of the moons dataset include:
- *noise*: changes the sharpness/blurriness of the two moons
- *angle*: changes the orientation of the moons relative to one another
- *sig_fraction*: changes the ratio of signal events to background events.
- *feature of merit (fom)*: The feature of merit for the moons dataset is the mass. Even though we will not be training on the feature of merit because this would eliminate the need for the relegation class altogether, it will still be useful as we calculate the significance of the signal sample. 
- *n_train_events*: changes the number of events that will be in the train dataset. For the moons dataset, this impacts the success of the relegation classifier. The classifier is most successful when n_train_events = 20000. 
- *hidden nodes*: changes the model architecture. 

### gen_master

`gen_master` begins by unpacking the dataset parameters and training parameters specified in the json string that gets passed in. This code is written for the moons dataset, so users with external data may have to make appropriate edits.

In the case of the moons dataset, the feature of merit follows a gaussian distribution. The following parameters characterize the gaussian and this is used in `make_peak_masks` in the relegator to subtract off the background events. 
``` python
min_fom = config_pars['data']['min_fom'] # 0.0
max_fom = config_pars['data']['max_fom'] # 1.0
mean_fom = config_pars['data']['mean_fom'] # 0.5
width_fom = config_pars['data']['width_fom'] # 0.03
n_sigmas = config_pars['data']['n_sigmas'] # 2.5

fom_name = 'm'

```
This example is set up so that a new dataset is generated with each run of `run_master_test.py`. For users bringing their own dataset, the code to generate a dataset can be omitted, and they can skip to the `else` statement below.

```python

if len(config_pars['data']['train_data_file']) == 0: #dataset has not been generated yet
    print('generating training dataset...')
    train_df = make_moons_mass(n_evts, min_fom, max_fom,
                               mean=mean_fom, sigma=width_fom,
                               noise=noise, angle=angle, beta=bkgd_beta)
else:
    print('unpickling training dataset from ' + config_pars['data']['train_data_file'])
    with open(config_pars['data']['train_data_file'], 'rb') as f:
        train_df = pickle.load(f)
```
We prepare the dataset for training by dropping the labels and splitting the data into train and test datasets. When initializing the model, we call `set_parameters` for the relegator. The user specifies the index of the signal, index of the background, the test and train data, as well as some features of the dataset and the feature of merit.  

``` python
releg.set_parameters(1, [0], sig_frac, test_frac, X_train, X_test, 'm', False, mean_fom, width_fom)

```
After converting `X_train`, `X_test`, `y_train`, and `y_test` to TensorFlow datasets, we generate peak masks that subtract off the background events using `relegator.gen_peak_masks()`.

Before the training begins, users can print a summary of their model that will look something like this:

![Alt](/summary.png "Summary")

After the model has finished training, gen_master contains some [plotting functionality](#plots) that you may find useful.

### relegator

Training begins in `gen_master.py` in the following way:
```python
releg.train(train_ds, test_ds, n_epochs, ot_cutoff, ot_cutoff_depth)
```
Once the training begins, the relegator class is autonomous in that the training proceeds without any user input. 

Note that we prevent overtraining by comparing `ot_cutoff` to `ot_cutoff_depth`.
```python 

if self.ot_cutoff and epoch + 1 > self.ot_cutoff_depth and loss_slope >= 0:
  break
```

## Plots

'gen_master.py' produces some basic plots of the results.

**Boundary Plot:** Here we can see the data being plotted, along with the boundaries that were drawn to classify the regions. The area shaded grey is the *relegated region*, in which these data points were excluded when drawing the boundary between the orange and blue moons. 

These results are not robust, as they vary greatly as the *noise* and *n_train_events* parameters change. In some runs, the relegator class does not behave as expected and omits a large portion of the data or does not omit any data at all.

![Alt](/noise_0.2_n_train_25000_run3.png "Boundary_Plot")

**Histogram:** The left plots show the gaussian distribution of the masses before the peak mask is applied. The right plots show the gaussian distribution after the peak mask is applied and we can see the increase in statistical significance of the signal sample.

![Alt](\masses_hist_successful.png "Masses_Histogram")

**Accuracy, Loss, and Significance vs. Epoch:** The training loop in `relegator.py` keeps lists of accuracy, loss, and significance for the train, test, and evaluation data for each epoch. In this plot we can see how these values evolved as the model trained.

![Alt](/stats_v_epoch.png "Stats")
