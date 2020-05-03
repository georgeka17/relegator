#!/usr/bon/env python
'''
runs master_moons_2.py with loops over parameters
'''

import os
import json
import numpy as np
import sys

ts = ['relegator'] #only testing relegator classifier here.
#, 'mod_binary']
# ts = ['relegator_factor'] #, 'mod_binary']
n_noise = 1
pow_range = (.1,1.1)
#only testing one noise value -- just want to see if the generalized relegator classifier works. 
#noises = np.linspace(pow_range[0], pow_range[1], n_noise + 1)

noises = [0.2]

n_trials = 1 #only run the model one time
n_train_events = 25000 #are we training in batches?
n_weighted_events = 100000 #is this the total? I don't see this show up anywhere in the code.

for n in noises:
    noise = n
    train_file_name = './datasets/train_ds_' + str(n_train_events)
    train_file_name += '_' + str(np.round(noise,4)) + '.pkl' #changed np.round(sig_frac,4 ) to np.round(noise,4)
    weight_file_name = './datasets/weighted_ds_' + str(n_weighted_events)
    weight_file_name += '_' + str(np.round(noise,4)) + '.pkl' #changed np.round(sig_frac,4 ) to np.round(noise,4)
    print(train_file_name)
    print(weight_file_name)
    for t in ts:
        # print(sig_frac)
        json_str = "{ \n \
        \"data\": { \n \
        \"n_events\": " + str(n_train_events) + ", \n \
        \"noise\": " + str(noise) + ", \n \
        \"angle\": 1.4, \n \
        \"sig_fraction\": 0.01, \n \
        \"test_fraction\": 0.25, \n \
        \"min_fom\": 0.0, \n \
        \"max_fom\": 1.0, \n \
        \"mean_fom\": 0.5, \n \
        \"width_fom\": 0.03, \n \
        \"n_sigmas\": 2.5, \n \
        \"bkgd_beta\": 0.6, \n \
        \"ttsplit_random_state\": 42, \n \
        \"weighted_n_events\": " + str(n_weighted_events) + ", \n \
        \"train_data_file\": \"" + train_file_name + "\", \n \
        \"weighted_data_file\": \"" + weight_file_name + "\" \n \
        }, \n \
        \"model\": { \n \
        \"model_type\": \""
        json_str += t
        json_str += "\", \n \
        \"input_dropout\": 0.05, \n \
        \"learning_rate\": 0.0005, \n \
        \"hidden_nodes\": [40, 40, 20], \n \
        \"bias\": true, \n \
        \"signif_type\": \"proba\" \n \
        }, \n \
        \"run\": { \n \
        \"n_epochs\": 1000, \n \
        \"ot_cutoff\": true, \n \
        \"ot_cutoff_depth\": 20 \n \
        } \n \
        }"
        # print(json_str)
        with open("config_run.json", "w") as f:
            f.write(json_str)
        if t == ts[0]:
            cmd = "python make_datasets_2_gen.py config_run.json"
            print(t, noise) #changed (t, sig_frac) to (t, noise)
            print(cmd)
            os.system(cmd)
        for j in range(n_trials):
            cmd = "python gen_master.py config_run.json" #noplot write_results"
            print(cmd)
            os.system(cmd)
