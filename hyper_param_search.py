import numpy as np
import os
import time
import random
import pandas as pd
import subprocess
from shutil import copyfile
# create 10 random values
def random_hyperparams(dropout_vec, lr_decay_vec, attention_dropout_vec, num_vals=10):
    return zip(np.around(np.random.uniform(low=dropout_vec[0],high=dropout_vec[1],size=num_vals),decimals=3),
            np.around(np.random.uniform(low=lr_decay_vec[0], high=lr_decay_vec[1], size=num_vals),decimals=3),
            np.around(np.random.uniform(low=attention_dropout_vec[0], high=attention_dropout_vec[1], size=num_vals),decimals=3))

# the process will call a function which will take up a chunk of hyper params
# and will call a function which can take a few
if __name__ == '__main__':
    filename = 'baseline_lstm.py'
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S").replace(" ","")
    out_dir_weights = os.path.join(os.getcwd(), cur_time)
    os.mkdir(out_dir_weights)

    params = random_hyperparams((0.5,0.5),(0.97,0.97),(0.5,0.5),num_vals=1)
    params = list(params)

    # new_params = random_hyperparams((0.6,0.5),(0.92,0.94),(0.5,0.5),num_vals=10)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.5, 0.8), (0.92, 0.94), (0.5, 0.8), num_vals=20)
    # params = params + list(new_params)
    # new_params = random_hyperparams((0.1, 0.5), (0.96, 0.96), (0.1, 0.6), num_vals=20)
    # params = params + list(new_params)


    #
    # new_params = random_hyperparams((0.53, 0.53), (0.95, 0.95), (0.23, 0.23), num_vals=1)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.58, 0.58), (0.95, 0.95), (0.41, 0.41), num_vals=1)
    # params = params + list(new_params)
    #
    #
    new_params = random_hyperparams((0.533, 0.533), (0.954, 0.954), (0.531, 0.531), num_vals=3)
    params = params + list(new_params)

    new_params = random_hyperparams((0.544, 0.544), (0.951, 0.951), (0.516, 0.516), num_vals=3)
    params = params + list(new_params)
    # new_params = random_hyperparams((0.3,0.5),(0.93,0.93),(0.3,0.80),num_vals=1)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.6,0.6),(0.97,0.97),(0.6,0.6),num_vals=1)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.75, 0.75), (0.97, 0.97), (0.75, 0.75), num_vals=1)
    # params = params + list(new_params)
    # new_params = random_hyperparams((0.75,0.75),(0.93,0.93),(0.75,0.75),num_vals=1)
    # params = params + list(new_params)
    #
    # # new_params = random_hyperparams((0.9, 0.9), (0.97, 0.97), (0.9, 0.9), num_vals=1)
    # # params = params + list(new_params)
    # # new_params = random_hyperparams((0.9,0.9),(0.93,0.93),(0.9,0.9),num_vals=1)
    # # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.85, 0.85), (0.97, 0.97), (0.85, 0.85), num_vals=1)
    # params = params + list(new_params)
    # new_params = random_hyperparams((0.85,0.85),(0.93,0.93),(0.85,0.85),num_vals=1)
    # params = params + list(new_params)
    #
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

    # new_params = random_hyperparams((0.6, 0.8), (0.95, 0.97), (0.6, 0.8), num_vals=10)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.3, 0.6), (0.92, 0.97), (0.3, 0.6), num_vals=10)
    # params = params + list(new_params)
    # new_params = random_hyperparams((0.1, 0.8), (0.92, 0.97), (0.1, 0.8), num_vals=10)
    # params = params + list(new_params)

    num_hidden_units_set = [192]
    num_hidden_units_list = [random.choice(num_hidden_units_set) for _ in range(len(params))]

    params = [x + (out_dir_weights,num_hidden_unit) for x,num_hidden_unit in zip(params,num_hidden_units_list)]
    pd.DataFrame(params,columns=['dropout','lr_decay','attention_dropout','out dir','hidden units']).\
                     to_csv(os.path.join(out_dir_weights,'hyperparam.log'),index=False, header=False)
    # copying the file to output dir for record
    copyfile(filename, os.path.join(out_dir_weights,filename))

    subprocess.call(['sh','baseline_lstm_bash.sh',cur_time + '/' + 'hyperparam.log',filename])


    # df = pd.DataFrame(params,columns=['dropout','lr_decay','attention_dropout','out dir','hidden units'])
    # df['Result'] = results
    # df.to_csv(os.path.join(out_dir_weights,'hyperparam.log'),index=False)
    # printing
    #
    # new_params = random_hyperparams((0.85,0.85),(0.93,0.93),(0.85,0.85),num_vals=1)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.75,0.75),(0.93,0.93),(0.75,0.75),num_vals=1)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.7,0.7),(0.93,0.93),(0.7,0.7),num_vals=1)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.5, 0.5), (0.93, 0.93), (0.5, 0.5), num_vals=1)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.8,0.8),(0.93,0.93),(0.8,0.8),num_vals=1)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.3,0.3),(0.93,0.93),(0.3,0.3),num_vals=1)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.4,0.4),(0.93,0.93),(0.4,0.4),num_vals=1)
    # params = params + list(new_params)
    #
    # new_params = random_hyperparams((0.2, 0.6), (0.92, 0.96), (0.2, 0.6), num_vals=50)
    # params = params + list(new_params)