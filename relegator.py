import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy import stats
import sys

class Relegator:
    def __init__(self): #setting up the model
        self.optimizer = None
        self.layers = None
        self.model = tf.keras.Sequential(self.layers)


        self.learning_rate = None
        self.hidden_layers_nodes = None
        self.bias = None
        self.n_inputs = 0
        self.n_outputs = 0 
        
        self.loss_object = None
        self.acc_object = tf.keras.metrics.CategoricalAccuracy()

        self.output_activation = 'softmax'
        self.name = 'relegation classifier'
        self.loss_type = 'rel. entropy + 1/sigma' 
    
    def build_model(self, nodes=[20,20,10], bias=True, n_ins=2, n_outs=1,
                    input_dropout=0.05): # setting up the model -- called by gen_master with parameters from json file.
        self.hidden_layers_nodes = nodes
        self.n_hidden_layers = len(self.hidden_layers_nodes)
        self.bias = bias
        self.n_inputs = n_ins
        self.n_outputs = n_outs
        self.input_dropout_frac = input_dropout
        self.layers = []

        if input_dropout > 0.0:
            self.layers.append(tf.keras.layers.Dropout(self.input_dropout_frac,
                                                       input_shape=(self.n_inputs, )))
            self.layers.append(tf.keras.layers.Dense(self.hidden_layers_nodes[0],
                                                     activation='relu',use_bias=self.bias))
        else:
            self.layers.append(tf.keras.layers.Dense(self.hidden_layers_nodes[0],
                                                     input_dim=self.n_inputs,
                                                     activation='relu', use_bias=self.bias))

        for i in range(self.n_hidden_layers - 1):
            self.layers.append(tf.keras.layers.Dense(self.hidden_layers_nodes[i+1],
                                                     activation='relu', use_bias=self.bias))

        self.layers.append(tf.keras.layers.Dense(self.n_outputs,
                                                 activation=self.output_activation))
        self.model = tf.keras.Sequential(self.layers)

    def init_optimizer(self, lr=1e-3):
        self.learning_rate = lr
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)

    def train_step(self, xs, y_truth, foms, mask, data_frac):
        with tf.GradientTape() as tape:
            y_pred = self.model(xs, training=True)
            loss_val, signif = Relegator.loss_object(self, y_truth, y_pred, foms, mask, data_frac)
            acc_val = self.acc_object(y_truth, y_pred)
        grads = tape.gradient(loss_val, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_val.numpy().mean(), acc_val.numpy().mean(), signif

    def predict_step(self, x, y_t, foms, mask, data_frac): 
        y_p = self.model(x, training=False)
        loss_val, signif= Relegator.loss_object(self, y_t, y_p, foms, mask, data_frac)
        loss_val = loss_val.numpy().mean()
        print(loss_val)
        acc_val = self.acc_object(y_t, y_p).numpy().mean()
        print(acc_val)
        return loss_val, acc_val, signif

    def train(self, train_ds, test_ds, max_epochs, ot_cutoff=True, ot_cutoff_depth=10): #These are the loops that the model goes through with the test and train data
        self.ot_cutoff = ot_cutoff
        self.ot_cutoff_depth = ot_cutoff_depth
        self.max_epochs = max_epochs
        epochs, eval_loss, eval_accs, eval_signif, train_loss, train_accs, train_signif, test_loss, test_accs, test_signif, test_acc_sma, test_loss_sma = [], [], [], [], [], [], [], [], [], [], [], []

        for epoch in range(self.max_epochs):
            epochs.append(epoch) #for future plots of accuracy, loss, and significance vs epoch
            lv, av = 0, 0
            for (batch, (xs, ys)) in enumerate(train_ds):
                lv, av, signif = self.train_step(xs, ys, self.train_foms, self.train_peak_mask, 1-self.test_fraction)
            print('Epoch {}/{} finished, learning rate: {:0.4f}'.format(epoch+1, self.max_epochs, self.optimizer.lr.numpy()))
            print('train loss: \t{:0.4f} \t|\ttrain acc: \t{:0.4f}'.format(lv, av))
            train_loss.append(lv) #for plots of loss vs epoch
            train_accs.append(av) #for plots of accuracy vs epoch
            train_signif.append(signif) #for plots of significance vs epoch
            for (batch, (xs, ys)) in enumerate(train_ds):
                lv, av, signif = self.predict_step(xs, ys, self.train_foms, self.train_peak_mask, 1-self.test_fraction)
            print('eval loss: \t{:0.4f} \t|\teval acc: \t{:0.4f}'.format(lv, av))
            eval_loss.append(lv) #for plots of loss vs epoch
            eval_accs.append(av) #for plots of accuracy vs epoch
            eval_signif.append(signif) #for plots of significance vs epoch
            for (batch, (xs, ys)) in enumerate(test_ds):
                lv, av, signif = self.predict_step(xs, ys, self.test_foms, self.test_peak_mask, self.test_fraction)
            print('test loss: \t{:0.4f} \t|\ttest acc: \t{:0.4f}'.format(lv, av))
            test_loss.append(lv) #for plots of loss vs epoch
            test_accs.append(av) #for plots of accuracy vs epoch
            test_signif.append(signif) #for plots of significance vs epoch
            print()

            loss_slope = 0
            epos = []
            if epoch + 1 > ot_cutoff_depth: #checking to prevent overtraining of the model
                epos = np.linspace(1, ot_cutoff_depth, ot_cutoff_depth)
                loss_slope, _, _, _, _ = stats.linregress(epos, test_loss[-ot_cutoff_depth:])

            if self.ot_cutoff and epoch + 1 > self.ot_cutoff_depth and loss_slope >= 0:
                break

        dict = {'eps': epochs,
                'eval_accs': eval_accs, 'eval_loss': eval_loss,
                'train_loss': train_loss, 'train_accs': train_accs,
                'test_loss': test_loss, 'test_accs': test_accs,
                'eval_signif': eval_signif,
                'train_signif': train_signif,
                'test_signif': test_signif}

            
        print('\nmodel trained for ' + str(len(epochs)) + ' epochs')
        print('final train accuracy:\t' + str(train_accs[-1]))
        print('final test accuracy:\t' + str(test_accs[-1]))
        self.train_results = train_results_df #results used by gen_master.py for plotting

    # # # # # # # # # # # # # # # # #

    def set_parameters(self, sig_idxs, bkgd_idxs, sig_frac, test_frac, train_ds, test_ds, fom_name = None, train_with_fom = False, fom_mean = 0, fom_width = 1e20): 
    	#One method that the user can call to set up all of their variables. Cleans things up on the front end.
        self.signal_idx = sig_idxs 
        self.background_idxs = bkgd_idxs

        self.train_foms = train_ds[fom_name]
        self.test_foms = test_ds[fom_name]

        train_ds.drop(fom_name, axis=1, inplace=True)
        test_ds.drop(fom_name, axis=1, inplace=True)

        self.fom_mean = fom_mean
        self.fom_width = fom_width
        self.fom_peak_range = (self.fom_mean - self.fom_width, self.fom_mean + self.fom_width)


        self.test_fraction = test_frac
        self.signal_fraction = sig_frac


    def gen_peak_masks(self): # this sets up a mask for both the train and test features of merit by calling make_peak_mask for both train and test.
        self.train_peak_mask = self.make_peak_mask(self.train_foms)
        self.test_peak_mask = self.make_peak_mask(self.test_foms)

    def make_peak_mask(self, foms):
        # gets the indices of events in the feature of merit peak
        # and then makes a mask array of 0s and 1s for dotting with feature of merit array
        peak_idxs = np.where(np.abs(foms-self.fom_mean) <= self.fom_width)
        peak_mask = np.zeros_like(foms)
        peak_mask[peak_idxs] = 1
        return peak_mask

    def signif_proba(self, y_truth, y_pred, foms, mask, data_frac):
        sig_mask  = tf.cast(tf.slice(y_truth, [0, self.signal_idx,], [len(y_truth), 1]), tf.float32)
        bkgd_mask = tf.cast(tf.slice(y_truth, [0, self.background_idxs[0],], [len(y_truth), 1]), tf.float32)
        sig_probs = tf.slice(y_pred, [0, self.signal_idx,], [len(y_truth), 1])
        peak_mask = tf.cast(mask, tf.float32)

        sig_as_sig_probs = tf.reshape(tf.math.multiply(sig_probs, sig_mask), (len(y_truth),))
        bkgd_as_sig_probs = tf.reshape(tf.math.multiply(sig_probs, bkgd_mask), (len(y_truth),))

        data_frac = tf.constant(data_frac)
        n_S = (1/data_frac) * tf.math.reduce_sum(tf.math.multiply(sig_as_sig_probs, peak_mask), axis=0)
        n_B = (1/data_frac) * tf.math.reduce_sum(tf.math.multiply(bkgd_as_sig_probs, peak_mask), axis=0)

        signif = self.signif_function(n_S, n_B, tf.constant(self.signal_fraction))
        return signif
        
    def signif_function(self, n_S, n_B, sig_frac):
        signif = tf.math.divide(n_S * sig_frac, tf.math.sqrt(n_S * sig_frac + n_B * (1 - sig_frac))) #calculation of the statistical significance
        return signif

    def loss_object(self, y_truth, y_pred, foms, mask, data_frac):
        signif = 0
        signif = self.signif_proba(y_truth, y_pred, foms, mask, data_frac)

        rel_ent = self.relegator_cce(y_truth, y_pred)
        return rel_ent + tf.math.divide(1, signif), signif #add the inverse of the significance so that significance is optimized in the loss function
        # return tf.keras.losses.categorical_crossentropy(y_truth, y_pred)

    def relegator_cce(self, y_truth, y_pred):
        y_p = []
        for i in range(self.n_outputs - 1):
            y_p.append(tf.transpose(tf.math.add(y_pred[:,i], y_pred[:,self.n_outputs-1])))
        y_p = tf.transpose(y_p)
        y_t = tf.slice(y_truth, [0, 0,], [len(y_truth), self.n_outputs - 1])
        return tf.keras.losses.categorical_crossentropy(y_t, y_p)
