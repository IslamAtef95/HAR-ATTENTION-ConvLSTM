import numpy as np
import os
import time
import pandas as pd
import subprocess
from shutil import copyfile
# create 10 random values
def random_hyperparams(dropout_vec, lr_decay_vec, attention_dropout_vec, num_vals=10):
    return zip(np.random.uniform(low=dropout_vec[0],high=dropout_vec[1],size=num_vals),
            np.random.uniform(low=lr_decay_vec[0], high=lr_decay_vec[1], size=num_vals),
            np.random.uniform(low=attention_dropout_vec[0], high=attention_dropout_vec[1], size=num_vals))

# the process will call a function which will take up a chunk of hyper params
# and will call a function which can take a few
if __name__ == '__main__':
    filename = 'baseline_longer_sliding_window.py'
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S").replace(" ","")
    out_dir_weights = os.path.join(os.getcwd(), cur_time)
    os.mkdir(out_dir_weights)

    params = random_hyperparams((0.65,0.65),(0.97,0.97),(0.65,0.65),num_vals=1)
    params = list(params)
    # new_params = random_hyperparams((0.6,0.6),(0.97,0.97),(0.6,0.6),num_vals=1)
    # params = params + list(new_params)

    new_params = random_hyperparams((0.75, 0.75), (0.97, 0.97), (0.75, 0.75), num_vals=1)
    params = params + list(new_params)

    new_params = random_hyperparams((0.80, 0.80), (0.97, 0.97), (0.80, 0.80), num_vals=1)
    params = params + list(new_params)


    # new_params = random_hyperparams((0.9, 0.9), (0.97, 0.97), (0.9, 0.9), num_vals=1)
    # params = params + list(new_params)
    # new_params = random_hyperparams((0.9,0.9),(0.93,0.93),(0.9,0.9),num_vals=1)
    # params = params + list(new_params)
    #
    new_params = random_hyperparams((0.85, 0.85), (0.97, 0.97), (0.85, 0.85), num_vals=1)
    params = params + list(new_params)
    # new_params = random_hyperparams((0.85,0.85),(0.93,0.93),(0.85,0.85),num_vals=1)
    # params = params + list(new_params)

    # new_params = random_hyperparams((0.3, 0.3), (0.97, 0.97), (0.3, 0.3), num_vals=1)
    # params = params + list(new_params)
    # new_params = random_hyperparams((0.35,0.35),(0.93,0.93),(0.35,0.35),num_vals=1)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.4, 0.4), (0.97, 0.97), (0.4, 0.4), num_vals=1)
    # params = params + list(new_params)
    # new_params = random_hyperparams((0.4,0.4),(0.93,0.93),(0.4,0.4),num_vals=1)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.5, 0.5), (0.97, 0.97), (0.5, 0.5), num_vals=1)
    # params = params + list(new_params)
    # new_params = random_hyperparams((0.5,0.5),(0.93,0.93),(0.5,0.5),num_vals=1)
    # params = params + list(new_params)

    new_params = random_hyperparams((0.75, 0.85), (0.96, 0.97), (0.7, 0.75), num_vals=20)
    params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.3, 0.6), (0.94, 0.97), (0.3, 0.6), num_vals=10)
    # params = params + list(new_params)

    # num_hidden_units_set = []
    # num_hidden_units_list = [random.choice(num_hidden_units_set) for _ in range(len(params))]

    params = [x + (out_dir_weights,) for x in params]
    pd.DataFrame(params,columns=['dropout','lr_decay','attention_dropout','out dir']).\
                     to_csv(os.path.join(out_dir_weights,'hyperparam.log'),index=False, header=False)
    # copying the file to output dir for record
    copyfile(filename, os.path.join(out_dir_weights,filename))

    subprocess.call(['sh','deep_conv_bash.sh',cur_time + '/' + 'hyperparam.log',filename])