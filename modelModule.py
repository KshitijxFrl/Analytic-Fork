#### importing nessary library ####

from pyexpat import model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler


##############################################


###### Data Input function ###################

def readCSV(inputCSV,x,y):
    Dataset = inputCSV
    Dataset = Dataset.iloc[:,x:y].values

    return Dataset


##############################################


###### Data Sequencer function ###################

def sequencer(data, seq_length):
    sc = MinMaxScaler()
    data = sc.fit_transform(data)
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)


    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))        

    return dataX,dataY



##############################################


###### Main Lstm module ###################


class LSTMnet(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers,seq_length):
        super(LSTMnet, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out


##############################################


###### Model Trainer ###################

def modelTrainer(data_x,data_y,seq_length):

    num_epochs = 10000
    learning_rate = 0.01

    input_size = 1
    hidden_size = 2
    num_layers = 1
    num_classes = 1
    seq_length = seq_length

    model = LSTMnet(num_classes, input_size, hidden_size, num_layers,seq_length)

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        outputs = model(data_x)
        optimizer.zero_grad()

        #obtain the loss function
        loss = criterion(outputs, data_y)
        loss.backward()
        optimizer.step()

    return model 



##############################################


###### Model Trainer ###################

def futurepred(steps,model,dataset):
    sc = MinMaxScaler()
    training_set3 = sc.fit_transform(dataset)
    a = 0
    box = []

    while a<steps:
        training_g = training_set3.reshape(-1,1)
        training_g = np.array(training_g[-4:])

        data_g = Variable(torch.Tensor(training_g.reshape(1,4,1)))
        output = model(data_g)
        
        training_set3 = np.append(training_set3,output.detach().numpy())
        box.append(sc.inverse_transform(output.detach().numpy()))
        #print(output)
        a = a+1

    return box     


if __name__ == "__main__":
    print("It is the model for LSTM time series prediction")