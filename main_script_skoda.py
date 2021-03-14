import argparse

import numpy as np
import _pickle as cp
from torch.optim.lr_scheduler import StepLR
import time
from hyper_param_search import random_hyperparams
from sliding_window import sliding_window
from preprocess_data import generate_data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import logging
import os
import torch.multiprocessing as multiprocessing
import pandas as pd
import scipy.io as sio

# dir = '/home/vishvak/Downloads/Skoda.mat'
dir = '/coc/pcba1/vmurahari3/deepconvlstmattention/Skoda.mat'

use_cuda = torch.cuda.is_available()

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 60

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 11

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 12
TEST_SLIDING_WINDOW_STEP = 1

# Batch Size
BATCH_SIZE = 100
# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 5
FILTER_STRIDE = 1

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

NUM_LAYERS_LSTM  = 2


def load_dataset(filename):

    contents = sio.loadmat(filename)
    X_train = contents['X_train']
    y_train = contents['y_train']
    X_test = contents['X_test']
    y_test = contents['y_test']

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

def massage_data():
    print("Loading data...")
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_dataset(dir)
    assert NB_SENSOR_CHANNELS == X_train_raw.shape[1]

    # Sensor data is segmented using a sliding window mechanism
    X_test, y_test = opp_sliding_window(X_test_raw, y_test_raw, SLIDING_WINDOW_LENGTH, TEST_SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))

    X_train, y_train = opp_sliding_window(X_train_raw, y_train_raw, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    print(" ..after sliding window (training): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
    return X_train,y_train,X_test,y_test, X_train_raw, y_train_raw, X_test_raw, y_test_raw

class RCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, is_bidirectional=False, dropout=0.5,attention_dropout=0.5):
        super(RCNN, self).__init__()
        self.is_bidirectional = is_bidirectional
        self.num_directions = 2 if is_bidirectional else 1
        self.hidden_size = hidden_size
        hidden_dim = hidden_size * self.num_directions
        self.dropout_val = dropout
        self.attention_dropout_val = attention_dropout
        self.conv2DLayer1 = nn.Conv2d(1, NUM_FILTERS, (FILTER_SIZE, 1), stride=(FILTER_STRIDE,1))
        self.relu1 = nn.ReLU()
        self.conv2DLayer2 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1), stride=(FILTER_STRIDE,1))
        self.relu2 = nn.ReLU()
        self.conv2DLayer3 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1), stride=(FILTER_STRIDE,1))
        self.relu3 = nn.ReLU()
        self.conv2DLayer4 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, (FILTER_SIZE, 1), stride=(FILTER_STRIDE,1))
        self.relu4 = nn.ReLU()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=is_bidirectional, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dense_layer = nn.Linear(hidden_dim , NUM_CLASSES)
        self.num_layers = num_layers
        # parameters for attention
        # self.W_Y_t = nn.Parameter(torch.mul(torch.randn(hidden_dim, hidden_dim),0.01))
        # self.W_h = nn.Parameter(torch.mul(torch.randn(hidden_dim, hidden_dim),0.01))
        # self.softmax_tranform = nn.Parameter(torch.mul(torch.randn(1, hidden_dim),0.01))
        self.attentionLayer1 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh1 = nn.Tanh()
        self.attentionLayer2 = nn.Linear(hidden_dim,1)
        self.softmax_attention = torch.nn.Softmax(dim=0)

    def forward(self, input, vis_attention=False):
        convout1 = self.conv2DLayer1(input)
        convout1 = self.relu1(convout1)
        convout2 = self.conv2DLayer2(convout1)
        convout2 = self.relu2(convout2)
        convout3 = self.conv2DLayer3(convout2)
        convout3 = self.relu3(convout3)
        convout4 = self.conv2DLayer4(convout3)
        convout4 = self.relu4(convout4)
        # reshape to put them in the lstm
        lstm_input = convout4.permute(2,0,1,3)
        lstm_input = lstm_input.contiguous()
        lstm_input = lstm_input.view(lstm_input.shape[0],lstm_input.shape[1],-1)
        # print(lstm_input.shape)
        # put things in lstm
        lstm_input = self.dropout(lstm_input)
        output, hidden = self.lstm(lstm_input, self.initHidden())
        # attention stuff
        past_context = output[:-1]
        current = output[-1]
        attention_layer1_output = self.attentionLayer1(past_context)
        # attention_layer1_output = attention_layer1_output + current
        attention_layer1_output = self.tanh1(attention_layer1_output)
        attention_layer1_output = self.attention_dropout(attention_layer1_output)
        attention_layer2_output = self.attentionLayer2(attention_layer1_output)
        attention_layer2_output = attention_layer2_output.squeeze(2)
        # find weights
        attn_weights = self.softmax_attention(attention_layer2_output)
        # the cols represent the weights
        attn_weights = attn_weights.unsqueeze(2)
        new_context_vector = torch.sum(attn_weights * past_context, 0)
        # use this new context vector for prediction
        # add a skip connection
        new_context_vector = new_context_vector + current
        logits = self.dense_layer(new_context_vector)
        if vis_attention:
            return logits, attn_weights
        return logits
        # output = output[-1] # considering only the last output
        # logits = self.dense_layer(output)
        # # we expect the size of the returned array to be something of size (batch size, out feats)
        # return logits
    # used at test time
    '''input : (batch_size,1,height,width)'''
    def predict(self,inputs, vis_attention=False):
        if vis_attention:
            logits,attn_weights = self.forward(inputs, vis_attention = vis_attention)
        else:
            logits = self.forward(inputs, vis_attention = vis_attention)
        # logits is of form ( batch, outfeats)
        softmax_layer = torch.nn.Softmax(dim=1)
        probs = softmax_layer(logits)
        _, idx = torch.max(probs, 1)
        if vis_attention:
            return idx, attn_weights
        else:
            return idx

    def initHidden(self):
        if use_cuda:
            h0 = Variable(torch.mul(torch.randn(self.num_layers * self.num_directions, BATCH_SIZE, self.hidden_size), 0.08)).cuda()
            c0 = Variable(torch.mul(torch.randn(self.num_layers * self.num_directions, BATCH_SIZE, self.hidden_size), 0.08)).cuda()
        else:
            h0 = Variable(torch.mul(torch.randn(self.num_layers * self.num_directions, BATCH_SIZE, self.hidden_size), 0.08))
            c0 = Variable(torch.mul(torch.randn(self.num_layers * self.num_directions, BATCH_SIZE, self.hidden_size), 0.08))

        return (h0,c0)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def evaluate(model, visualize_attention=False, out_dir= None, pred_dir = None):
    # Classification of the testing data
    final_weights = []
    y_test_raw = data.y_test_raw
    X_test = data.X_test
    y_test = data.y_test
    model.eval()
    print("Processing {0} instances in mini-batches of {1}".format(X_test.shape[0], BATCH_SIZE))
    test_pred = np.zeros_like(y_test_raw)
    test_true = y_test_raw
    start = SLIDING_WINDOW_LENGTH - 1
    for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE):
        inputs, targets = batch
        inputs = Variable(torch.from_numpy(inputs),volatile=True)
        # targets = torch.from_numpy(targets)
        inputs = inputs.unsqueeze(1)  # adding an extra singleton dimension for input to CNN
        if use_cuda:
            inputs = inputs.cuda()
            # targets = targets.cuda()
        if visualize_attention:
            y_pred, attn_weights = model.predict(inputs,vis_attention=visualize_attention)
            attn_weights = attn_weights.squeeze(2)
            attn_weights = attn_weights.data.cpu().numpy()
            attn_weights = np.transpose(attn_weights)
            attn_list = attn_weights.tolist()
            final_weights = final_weights + attn_list
        else:
            y_pred = model.predict(inputs)
        y_pred = y_pred.data.cpu().numpy()
        test_pred[start: (start + targets.shape[0])] = y_pred
        start = start + targets.shape[0]
        # test_true = np.append(test_true, targets, axis=0)
    # Results presentation

    print("||Results||")
    import sklearn.metrics as metrics
    score = metrics.f1_score(test_true, test_pred, average='macro')
    print("\tTest fscore:\t{:.4f} ".format(score))

    if visualize_attention:
        # write to file the attn weights
        # first adding dummy attn values for the first sliding_window_length - 1 values
        dummy_weights = [[0]*len(final_weights[0]) for _ in range(SLIDING_WINDOW_LENGTH-1)]
        final_weights = dummy_weights + final_weights
        dummy_weights = [[0]*len(final_weights[0]) for _ in range(test_pred.shape[0]-len(final_weights))]
        final_weights = final_weights + dummy_weights
        final_df = pd.DataFrame(final_weights)
        final_df['prediction'] = test_pred
        final_df['ground_truth'] = test_true
        final_df.to_csv(out_dir)

        pred_df = pd.DataFrame(np.hstack((test_pred.reshape((-1,1)), test_true.reshape(-1,1))))
        pred_df.to_csv(pred_dir, index=False)

        # also write some extra stats

    return score

def train_model(model,output_dir, num_epochs= 5, lr=1e-3, lr_decay = 0.97, momentum = 0.9, print_every = 80, evaluate_every=2, handle=None):
    import copy
    X_train = data.X_train
    y_train = data.y_train
    # cur_momentum = 0.5
    optimizer = optim.RMSprop(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=2, gamma=lr_decay)
    best_score = 0
    best_weights = None
    best_model = None
    for epoch in range(num_epochs):
        model.train("True")
        for iter, batch in enumerate(iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True)):
            optimizer.zero_grad()
            inputs,targets = batch
            # convert to variables
            inputs = Variable(torch.from_numpy(inputs))
            targets = Variable(torch.from_numpy(targets))
            inputs = inputs.unsqueeze(1) # adding an extra singleton dimension for input to CNN
            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
            targets = targets.long() # casting to long for the loss function
            output = model(inputs)
            loss = criterion(output,targets)
            if iter % print_every == 0:
                if handle != None:
                   handle.write("train loss {0}, epoch: {1} ,iteration: {2} \n".format(loss.data[0], epoch, iter))
                print("train loss {0}, epoch: {1} ,iteration: {2} ".format(loss.data[0], epoch, iter))
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()

        if (epoch + 1) % evaluate_every == 0:
            score = evaluate(model)
            if score > best_score:
                best_score = score
                best_weights = model.state_dict()
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), os.path.join(output_dir, str(best_score) + ".." + str(model.attention_dropout_val) + ".."
                                                                 + str(model.dropout_val) + ".." + str(lr_decay)))

            if handle != None:
                handle.write("score : {0} \n".format(score))
                handle.write("best score: {0} \n".format(best_score))
            print("best score: {0}".format(best_score))

        print(best_score)
        if (epoch + 1) == 40:
            if best_score < 0.64:
                # these are trash params
                return best_score, best_model

        # if (epoch + 1) % 20 == 0:
        #     # change momentum
        #     cur_momentum += 0.08
        #     if cur_momentum > momentum:
        #         cur_momentum = momentum
        #     optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=cur_momentum)

        scheduler.step()

    return best_score,best_model

class DataWrapper:
    def __init__(self):
       self.X_train, self.y_train, self.X_test, self.y_test,self.X_train_raw, \
                self.y_train_raw, self.X_test_raw, self.y_test_raw = massage_data()

data = DataWrapper()


def read_cmd_args():
    ad = 0.7
    d = 0.6
    lr_decay = 0.95
    hunits = 0
    out_dir = None
    # output_file = None
    parser = argparse.ArgumentParser(description='LSTM attention models.')
    parser.add_argument("-id", "--identity", type=int)
    parser.add_argument("-ad", "--attention_dropout", type=float)
    parser.add_argument('-d', "--dropout", type=float)
    parser.add_argument("-lr_decay", "--lr_decay", type=float)
    # parser.add_argument("-units", "--hidden", type=int)
    parser.add_argument("-out", "--output_dir", type=str)
    # # logging all input params
    args = parser.parse_args()
    if args.attention_dropout:
        ad = args.attention_dropout
    if args.dropout:
        d = args.dropout
    if args.lr_decay:
        lr_decay = args.lr_decay
    # if args.hidden:
    #     hunits = args.hidden
    if args.output_dir:
        out_dir = args.output_dir

    model = RCNN(input_size=NB_SENSOR_CHANNELS*NUM_FILTERS, hidden_size=NUM_UNITS_LSTM,
                 num_layers=NUM_LAYERS_LSTM, is_bidirectional=False, dropout=d, attention_dropout=ad)
    if use_cuda:
        model = model.cuda()
    # best_score, weights = train_model(model, lr_decay=lr_decay, num_epochs=120)
    # torch.save(weights, os.path.join(out_dir, str(best_score) + ".." + str(ad) + ".."
    #                                  + str(d) + ".." + str(lr_decay)) + ".." + str(hunits))
    best_score, best_model = train_model(model,out_dir, lr_decay=lr_decay, num_epochs=200)
    # torch.save(best_model.state_dict(),os.path.join(out_dir, str(best_score) + ".." + str(ad) + ".."
    #                                  + str(d) + ".." + str(lr_decay)) + ".." + str(hunits))

    return best_score

# train model
if __name__ == '__main__':
    # model = RCNN(input_size=NB_SENSOR_CHANNELS * NUM_FILTERS, hidden_size=NUM_UNITS_LSTM,
    #              num_layers=NUM_LAYERS_LSTM, is_bidirectional=False, dropout=0.5, attention_dropout=0.5)
    # evaluate(model,visualize_attention=True, out_dir='/home/vishvak/LargeScaleFeatureLearning'
    #                         '/DeepConvLSTM/attn_weights.csv', pred_dir='/home/vishvak/LargeScaleFeatureLearning'
    #                         '/DeepConvLSTM/preds.csv')
    # model = modelload()
    # evaluate(model,visualize_attention=True,out_dir=)

    read_cmd_args()

    # read command line arguments
    # ad = 0.7
    # d = 0.6
    # lr_decay = 0.95
    # output_file = None
    # parser = argparse.ArgumentParser(description='LSTM attention models.')
    # parser.add_argument("-ad","--attention_dropout",type=float)
    # parser.add_argument('-d', "--dropout",type=float)
    # parser.add_argument("-lr_decay", "--lr_decay",type=float)
    # parser.add_argument("-ofile", "--output_file")
    # # logging all input params
    # args = parser.parse_args()
    # if args.attention_dropout:
    #     ad = args.attention_dropout
    # if args.dropout:
    #     d = args.dropout
    # if args.lr_decay:
    #     lr_decay = args.lr_decay
    # if args.output_file:
    #     output_file = args.output_file
    # data = DataWrapper()
    # print(args.output_file)
    # output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), output_file)
    # print(output_file)
    # # setting up output file
    # handle = open(output_file,"w+")
    # handle.write("dropout: {0} , lr_decay: {1} , attention_dropout: {2}  \n".format(d, lr_decay, ad))
    # model = RCNN(input_size=NB_SENSOR_CHANNELS * NUM_FILTERS, hidden_size=NUM_UNITS_LSTM,
    #              num_layers=NUM_LAYERS_LSTM,is_bidirectional=False,dropout=0.6,attention_dropout=0.6)
    # if use_cuda:
    #     model = model.cuda()
    # print("train_model reached")
    #
    # a_num = np.random.randint(1,100000)
    # a = open('hell{0}.txt'.format(a_num),'w+')
    # a.close()
    # train_model(model,lr_decay=0.97,num_epochs=110)
    # handle.close()
    # out_dir_weights = os.path.join(os.getcwd(), time.strftime("%Y-%m-%d %H:%M:%S").replace(" ",""))
    # os.mkdir(out_dir_weights)
    # params = random_hyperparams((0.6,0.6),(0.93,0.93),(0.6,0.6),num_vals=1)
    # params = list(params)
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
    # # new_params = random_hyperparams((0.2, 0.5), (0.97, 0.97), (0.2, 0.5), num_vals=5)
    # # params = params + list(new_params)
    #
    # params = [x + (out_dir_weights,) for x in params]
    # pd.DataFrame(params,columns=['dropout','lr_decay','attention_dropout','out dir']).\
    #                  to_csv(os.path.join(out_dir_weights,'hyperparam.log'),index=False)
    # print([dropout,lr_decay,attention_dropout] for dropout,lr_decay,attention_dropout in params)
    # multiprocessing.set_start_method('spawn')
    # pool = multiprocessing.Pool(processes=3)
    # results = pool.map(run_model, params)
    # # printing out accuracies
    #
    # pool.close()
    # print(results[:])
    # df = pd.DataFrame(params,columns=['dropout','lr_decay','attention_dropout','out dir'])
    # df['Result'] = results
    # df.to_csv(os.path.join(out_dir_weights,'hyperparam.log'),index=False)
    # # printing params
    #

