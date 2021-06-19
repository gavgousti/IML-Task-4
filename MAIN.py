import numpy as np
import cv2             
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import pandas as pd
import torch 
from torch import nn
from tqdm.notebook import tqdm
import torch.nn.functional as F


def Preprocessing():    # at this function we take embedings for the 10000 images (it takes long to run).
    folder = r'C:\Users\41778\Desktop\task4\food' # add here the path of folder food
    model = tf.keras.applications.ResNet50(include_top= False, weights="imagenet", pooling= 'avg')
    X = np.zeros((10000, 2048)) ; i = 0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None: 
            img = img.astype('float64')            
            img = np.expand_dims(img, axis = 0)
            feature = model(img)
            X[i,:] = np.reshape(feature, 2048)
            i += 1
            if i % 500 ==0:
                print('{:3.0f}%'.format(i/100))
    
    pd.DataFrame(X).to_csv("ResNet50.csv", index = False, header = None)

def main():

    def predict(model, loader, save): #function for making pradictions
        model.eval()
        curr = 0
        prediction = np.zeros(len(loader.dataset)) 
        for batch_idx, data in enumerate(tqdm(loader, desc="Predicting", leave=False)):
            data1 = data[:,0:ncols]
            data2 = data[:,ncols:2*ncols]
            data3 = data[:,2*ncols:3*ncols]
            optimizer.zero_grad()
            anchor = model(data1)
            positive = model(data2)
            negative = model(data3)      
            anchor = to_unit_vector(anchor)
            positive = to_unit_vector(positive)
            negative = to_unit_vector(negative)
            dist_a = F.pairwise_distance(anchor, positive, 2)
            dist_b = F.pairwise_distance(anchor, negative, 2)
            log = (dist_a < dist_b).type(torch.uint8)
            prediction[curr:curr+len(data1)] = log
            curr += len(data1)
        if save:
            pd.DataFrame(prediction).to_csv("Prediction.csv", index = False, header = None)  
        return prediction

    def FastAcc(model, loader): #function for taking accuracy (we dont use it here, but it was during grid search)
        pred = predict(model, loader, save=False)
        acc = np.mean(pred)
        return acc

    def fix_indexes(ar): #taking a nice matrix with the indeces of the photos for every triplet
        ind = np.zeros((len(ar), 3),  dtype=np.int16 )
        for i in range(len(ar)):
            ind[i,0] = int(ar[i][0][0:5])
            ind[i,1] = int(ar[i][0][6:11])
            ind[i,2] = int(ar[i][0][12:])
        return ind

    def fix_acording_to_triplet(triplet, Xtotal): #take feature matrix for training
        ncols = Xtotal.shape[1]
        X = np.zeros((len(triplet), 3*ncols))
        for i in range(len(triplet)):
            X[i, 0:ncols] = Xtotal[triplet[i, 0]]
            X[i, ncols:2*ncols] = Xtotal[triplet[i, 1]]
            X[i, 2*ncols:3*ncols] = Xtotal[triplet[i, 2]]
        return X

    class NeuralNetwork(nn.Module):

        def __init__(self):
            super(NeuralNetwork, self).__init__()

            self.net = nn.Sequential(
                nn.Dropout(p=0.75),
                nn.Linear(ncols, 600),
                nn.BatchNorm1d(600),
                nn.PReLU(),
                nn.Linear(600, 300),
                nn.BatchNorm1d(300),
                nn.PReLU()    
            )
        
        def forward(self, x):
            logits = self.net(x)
            return logits    
    
    def train_loop(dataloader, model, loss_fn, optimizer):
        for batch, data in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            anchor = data[:,0:ncols]
            positive = data[:,ncols:2*ncols]
            negative = data[:,2*ncols:3*ncols]
            data1 = to_unit_vector(model(anchor))
            data2 = to_unit_vector(model(positive))
            data3 = to_unit_vector(model(negative))
            loss = loss_fn(data1, data2, data3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
            loss = loss.item()
        print(f"loss:  {loss:1.5f}")  

    def to_unit_vector(vec):
        return vec/(torch.linalg.norm(vec, dim=1, ord=2).unsqueeze(-1))

    print(f"=====================\nPREPROCESSING STARTED\n=====================")

    Xtotal = pd.read_csv("ResNet50.csv", header = None).to_numpy()
    ncols = Xtotal.shape[1]

    train_indexes = pd.read_csv('train_triplets.txt', header = None).to_numpy()
    test_indexes = pd.read_csv('test_triplets.txt', header = None).to_numpy()
    train_indexes = fix_indexes(train_indexes)
    test_indexes = fix_indexes(test_indexes)

    X_big_test = torch.from_numpy(fix_acording_to_triplet(test_indexes, Xtotal)).float()
    X_big_train = torch.from_numpy(fix_acording_to_triplet(train_indexes, Xtotal)).float()

    print(f"=====================\nPREPROCESSING ENDED\n=====================")
    print(f"=====================\nTRAINING STARTED\n=====================")



    trainloader = torch.utils.data.DataLoader(X_big_train, shuffle=True, batch_size=200)
   
    model = NeuralNetwork()
    print(model)

    print("Model Parameters: {}".format(sum(p.numel() for p in model.parameters())))
    margin = 1
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    learning_rate = 1e-3
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"=====================\nEPOCH {t+1}\n=====================")
        train_loop(trainloader, model, loss_fn, optimizer)

    test_loader_final  = torch.utils.data.DataLoader(X_big_test, shuffle=False, batch_size=200)

    print(f"=====================\nTRAINING ENDED WAIT A BIT FOR THE PREDICTION\n=====================")


    predict(model, test_loader_final, save=True)

###################################################################################################################

Preprocessing()       #VERY IMPORTANT for you to run this function in order to take ResNet50.csv file (it takes long)
main()