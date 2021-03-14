import argparse

import numpy as np
import _pickle as cp
from torch.optim.lr_scheduler import StepLR
import time
from hyper_param_search import random_hyperparams
from sliding_window import sliding_window
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import os
import torch.multiprocessing as multiprocessing
import pandas as pd

use_cuda = torch.cuda.is_available()

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 77

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 384

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
FILTER_STRIDE = 2
# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

NUM_LAYERS_LSTM  = 2

# data_dir =  "/home/vishvak/LargeScaleFeatureLearning"
# out_dir = "/home/vishvak/LargeScaleFeatureLearning/OpportunityUCIDataset/dataset/oppChallenge_gestures.data"
data_dir = "/coc/pcba1/Datasets/public"
out_dir = "oppChallenge_gestures.data"

def load_dataset(filename):

    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

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
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_dataset(out_dir)
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

        # self.attentionLayer1 = nn.Linear(hidden_dim, hidden_dim)
        # self.tanh1 = nn.Tanh()
        # self.attentionLayer2 = nn.Linear(hidden_dim,1)
        # self.softmax_attention = torch.nn.Softmax(dim=0)

    def forward(self, input):
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
        # put things in lstm
        # lstm_input = self.dropout(lstm_input)
        output, hidden = self.lstm(lstm_input, self.initHidden())
        # attention stuff
        past_context = output[:-1]
        current = output[-1]
        #
        # attention_layer1_output = self.attentionLayer1(past_context)
        # attention_layer1_output = attention_layer1_output + current
        # attention_layer1_output = self.tanh1(attention_layer1_output)
        # attention_layer1_output = self.attention_dropout(attention_layer1_output)
        # attention_layer2_output = self.attentionLayer2(attention_layer1_output)
        # attention_layer2_output = attention_layer2_output.squeeze(2)
        # find weights
        # attn_weights = self.softmax_attention(attention_layer2_output)
        # the cols represent the weights
        # attn_weights = attn_weights.unsqueeze(2)
        # new_context_vector = torch.sum(attn_weights * past_context, 0)
        # use this new context vector for prediction
        # add a skip connection
        # new_context_vector = new_context_vector + current
        new_context_vector = current
        # new_context_vector = self.attention_dropout(new_context_vector)
        logits = self.dense_layer(new_context_vector)
        return logits
        # output = output[-1] # considering only the last output
        # logits = self.dense_layer(output)
        # # we expect the size of the returned array to be something of size (batch size, out feats)
        # return logits
    # used at test time
    '''input : (batch_size,1,height,width)'''
    def predict(self,inputs):
        logits = self.forward(inputs)
        # logits is of form ( batch, outfeats)
        softmax_layer = torch.nn.Softmax(dim=1)
        probs = softmax_layer(logits)
        _, idx = torch.max(probs, 1)
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

def evaluate(model):
    # Classification of the testing data
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
        inputs = Variable(torch.from_numpy(inputs), volatile=True)
        # targets = torch.from_numpy(targets)
        inputs = inputs.unsqueeze(1)  # adding an extra singleton dimension for input to CNN
        if use_cuda:
            inputs = inputs.cuda()
            # targets = targets.cuda()
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
    return score

def train_model(model, num_epochs= 5, lr=1e-3, lr_decay = 0.97, momentum = 0.9, print_every = 40, evaluate_every=2, handle=None):
    import copy
    X_train = data.X_train
    y_train = data.y_train
    # cur_momentum = 0.5
    optimizer = optim.RMSprop(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=lr_decay)
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
            torch. nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()

        if (epoch + 1) % evaluate_every == 0:
            score = evaluate(model)
            if score > best_score:
                best_score = score
                best_weights = model.state_dict()
                best_model = copy.deepcopy(model)
            if handle != None:
                handle.write("score : {0} \n".format(score))
                handle.write("best score: {0} \n".format(best_score))
            print("best score: {0}".format(best_score), )
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
    # hunits = 0
    out_dir = None
    # output_file = None
    parser = argparse.ArgumentParser(description='LSTM attention models.')
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

    model = RCNN(input_size=NB_SENSOR_CHANNELS * NUM_FILTERS, hidden_size=NUM_UNITS_LSTM,
                 num_layers=NUM_LAYERS_LSTM, is_bidirectional=False, dropout=d, attention_dropout=ad)
    if use_cuda:
        model = model.cuda()
    best_score, best_model = train_model(model, lr_decay=lr_decay, num_epochs=108)
    torch.save(best_model.state_dict(), os.path.join(out_dir, str(best_score) + ".." + str(ad) + ".."
                                     + str(d) + ".." + str(lr_decay)))
    return best_score

# train model
if __name__ == '__main__':
    read_cmd_args()
