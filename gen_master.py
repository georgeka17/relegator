#!/usr/bon/env python
'''
Fits arbitrary model to moons+mass data.  Command line is

python master_moons.py <config_file>.json

model can take the values: regress, nn_binary, relegator

Last three arguments may be omitted to run with default values.
'''

from colorama import Fore, Back, Style
import pandas as pd
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import sys, os
import pickle
import json
import time

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf

from moons_tools_2 import *
from relegator import *

# check if tf2.0 easr exec is working
print('Executing eagerly today -->  ' + str(tf.executing_eagerly()) + '!\n')
if not tf.executing_eagerly():
    sys.exit('this code works with tf2.0+ (eager execution) only')

from matplotlib import rc

#unpacking information about the dataset from the json configuration file.

with open(sys.argv[1], 'r') as f:
    config_pars = json.loads(f.read())
model_type = config_pars['model']['model_type']
allowed_models = ['regress', 'binary_softmax', 'relegator',
                  'mod_binary', 'relegator_factor', 'relegator_diff']
if model_type not in allowed_models:
    error_message = str('error: model type \"' + model_type + '\" undefined')
    sys.exit(error_message)

n_evts = config_pars['data']['n_events'] #20000
noise = config_pars['data']['noise'] # 0.3
angle = config_pars['data']['angle'] # 0.0
test_frac = config_pars['data']['test_fraction'] # 0.25
sig_frac = config_pars['data']['sig_fraction'] # 0.5

bkgd_beta = config_pars['data']['bkgd_beta'] # 0.6
ttsplit_random_state = config_pars['data']['ttsplit_random_state'] # 42

# parameters for 'mass' distribution
min_fom = config_pars['data']['min_fom'] # 0.0
max_fom = config_pars['data']['max_fom'] # 1.0
mean_fom = config_pars['data']['mean_fom'] # 0.5
width_fom = config_pars['data']['width_fom'] # 0.03
n_sigmas = config_pars['data']['n_sigmas'] # 2.5

fom_name = 'm'

train_df = None
if len(config_pars['data']['train_data_file']) == 0: #dataset has not been generated yet
    print('generating training dataset...')
    train_df = make_moons_mass(n_evts, min_fom, max_fom,
                               mean=mean_fom, sigma=width_fom,
                               noise=noise, angle=angle, beta=bkgd_beta)
else: #dataset was already generated and we can load it in as a dataframe
    print('unpickling training dataset from ' + config_pars['data']['train_data_file'])
    with open(config_pars['data']['train_data_file'], 'rb') as f:
        train_df = pickle.load(f)
foms = train_df[[fom_name]]

y = train_df['label']
y_1hot = pd.concat([train_df['label_0'], train_df['label_1']], axis=1, sort=False)
if 'relegator' in model_type:
    y_1hot['label_rel'] = 0
    y = y_1hot.copy()

#unpacking relegator training specifications from the json configuration file.

input_dropout = config_pars['model']['input_dropout'] # 0.05
learning_rate = config_pars['model']['learning_rate'] # 1e-3
hidden_nodes = config_pars['model']['hidden_nodes'] # [40, 40, 20]
bias = config_pars['model']['bias'] # True
signif_type = config_pars['model']['signif_type'] # True

n_epochs = config_pars['run']['n_epochs'] # 100
ot_cutoff_depth = config_pars['run']['ot_cutoff_depth'] # 20
ot_cutoff = config_pars['run']['ot_cutoff'] # True

releg = Relegator()

#prepare the dataset for training -- drop labels, train/test split

train_df.drop(['label', 'label_0', 'label_1'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=test_frac)

#identifying the feature of merit
masses_train = X_train['m']
masses_test = X_test['m']

#set up parameters for the model to train on.
releg.set_parameters(1, [0], sig_frac, test_frac, X_train, X_test, 'm', False, mean_fom, width_fom)
#classification goes as: [background, signal, relegated]

n_inputs = len(X_train.columns)
n_outputs = len(y_train.columns) # number of inputs + 1, to index the relegated data.

#numpy to tensorflow dataset.
train_ds = np_to_tfds(X_train, y_train, batch_size=len(X_train)) 
test_ds = np_to_tfds(X_test, y_test, batch_size=len(X_test))

releg.gen_peak_masks()

releg.build_model(hidden_nodes, bias, n_inputs, n_outputs, input_dropout)

releg.model.summary()

releg.init_optimizer(learning_rate)

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test  = X_test.to_numpy()
y_test  = y_test.to_numpy()

start_time = time.time()
releg.train(train_ds, test_ds, n_epochs, ot_cutoff, ot_cutoff_depth)
train_results_df = releg.train_results
train_time = time.time() - start_time

print('\n... NN trained, plotting...\n')


fig = plt.figure(figsize=(9,6))
nbins = int(np.sqrt(n_evts)/2)

n_rows, n_cols = 2, 2
ax = plt.subplot(n_rows,n_cols, 1)
plt.plot(train_results_df['eps'], train_results_df['train_accs'], label='train, dropout (' + str(input_dropout) + ')')
plt.plot(train_results_df['eps'], train_results_df['eval_accs'], label='train dataset')
plt.plot(train_results_df['eps'], train_results_df['test_accs'], label='test dataset')
plt.title('model accuracy')
plt.legend(loc='lower right')
plt.ylabel('accuracy')
plt.xlabel('epoch')

y_pred_train, y_pred_test = [], []

ax = plt.subplot(n_rows, n_cols, 3)
opt_thr = 0.0

#decision boundary plots
fig = plt.figure(figsize=(9,5.5))
ax = plt.subplot(1,1,1)
x1_mesh, x2_mesh, class_mesh = predict_bound_class(releg.model, train_df, releg.n_outputs) #, opt_thr=opt_thr)
# custom colormap for contour
vmin, vmax = 0, 2
rel_cmap = relegator_cmap()
plt.register_cmap(cmap=rel_cmap)
cont = ax.contourf(x1_mesh, x2_mesh, class_mesh, alpha=0.45, cmap='RelegatorCMap', vmin=vmin, vmax=vmax, levels=[-0.5, 0.5, 1.5, 2.5])

plot_xs(X_train, y_train[:,1], ax)
plt.title('noise = ' + str(noise) + ', angle = ' + str(angle) + ', epochs = ' + str(n_epochs))
plt.tight_layout()

fig = plt.figure(figsize=(9,5.5))
ax = plt.subplot(1,1,1)
cont = ax.contourf(x1_mesh, x2_mesh, class_mesh, alpha=0.5, cmap='RelegatorCMap', vmin=vmin, vmax=vmax, levels=[-0.5, 0.5, 1.5, 2.5])
plt.title('noise = ' + str(noise) + ', angle = ' + str(angle) + ', epochs = ' + str(n_epochs))
plt.tight_layout()

print('\napplying optimal cut to dataset with sig_frac = ' + str(sig_frac) + '...')

weighted_n_evts, weighted_df = 0, None
with open(config_pars['data']['weighted_data_file'], 'rb') as f:
  weighted_df = pickle.load(f)
  weighted_n_evts = len(weighted_df.index)

y_weighted = weighted_df['label']
xs_weighted = weighted_df.drop(['label', 'label_0', 'label_1', fom_name], axis=1)

fig = plt.figure(figsize=(11,6))
ax = plt.subplot2grid((3, 2), (0, 0), rowspan=2)
cents, occs, bkgds = hist_ms(weighted_df, min_fom, max_fom, nbins, ax,
                             sig_limits=(mean_fom - n_sigmas*width_fom, mean_fom + n_sigmas*width_fom))
plt.xlim((min_fom, max_fom))
plt.title('masses, sig_frac = ' + str(sig_frac))
plt.legend(loc='upper right')

ax = plt.subplot2grid((3, 2), (2, 0))
ax.yaxis.grid(True)
# hist_diff_signif(cents, occs, bkgds)
# plt.ylim((-10, 10))
hist_residuals(cents, occs, bkgds,
               sig_limits=(mean_fom - n_sigmas*width_fom, mean_fom + n_sigmas*width_fom))
plt.xlim((min_fom, max_fom))

ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)
raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig = 0, 0, 0, 0, 0, 0
cents, occs, bkgds = 0, 0, 0

weighted_df['prob_0'] = releg.model.predict(xs_weighted)[:,0]
weighted_df['prob_1'] = releg.model.predict(xs_weighted)[:,1]
weighted_df['prob_rel'] = releg.model.predict(xs_weighted)[:,-1]
cents, occs, bkgds = hist_softmax_cut_ms(weighted_df, min_fom, max_fom, nbins, ax, sig_limits=(mean_fom - n_sigmas*width_fom, mean_fom + n_sigmas*width_fom))
raw_signif, pass_signif, n_raw_bkgd, n_raw_sig, n_pass_bkgd, n_pass_sig = compute_signif_binary(weighted_df, mean_fom, width_fom, n_sigmas)
plt.xlim((min_fom, max_fom))

title_str = 'post-cut masses'
if opt_thr > 0:
    title_str += ', opt. threshold = %0.3f' % opt_thr
plt.title(title_str)
plt.legend(loc='upper right')

ax = plt.subplot2grid((3, 2), (2, 1))
ax.yaxis.grid(True)

hist_residuals(cents, occs, bkgds, sig_limits=(mean_fom - n_sigmas*width_fom, mean_fom + n_sigmas*width_fom))
plt.xlim((min_fom, max_fom))

plt.tight_layout()

print('\nraw analysis significance:\t', str(raw_signif))
print('pass analysis significance:\t', str(pass_signif))

plt.show()
