# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:40:53 2020

@author: Kripa

"""
import numpy as np
n_noise = 10
pow_range = (.1,1.1)
noises = np.linspace(pow_range[0], pow_range[1], n_noise + 1)
print(noises)
