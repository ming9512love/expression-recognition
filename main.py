from __future__ import print_function
import argparse
import numpy  as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from data_loaders import Plain_Dataset, eval_data_dataloader
from deep_emotion import Deep_Emotion
from generate_data import Generate_data

from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

def Train(epochs,train_loader,val_loader,criterion,optmizer,device):
    y_true = []
    y_pred = []
    '''
    Training Loop
    '''
    print("===================================Start Training===================================")
    for e in range(epochs):
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0
        # Train the model  #
        net.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optmizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs,labels)
            loss.backward()
            optmizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs,1)
            train_correct += torch.sum(preds == labels.data)

        #validate the model#
        net.eval()
        for data,labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = net(data)
            val_loss = criterion(val_outputs, labels)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs,1)
            val_correct += torch.sum(val_preds == labels.data)
            if (e == epochs -1):

                val_pred = val_preds.cpu().numpy().tolist()
                y_pred += val_pred
                label = labels.data.cpu().numpy().tolist()
                y_true += label
   

        train_loss = train_loss/len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        validation_loss =  validation_loss / len(validation_dataset)
        val_acc = val_correct.double() / len(validation_dataset)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc*100)
        val_loss_list.append(validation_loss)
        val_acc_list.append(val_acc*100)
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}%'
                                                           .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100))

    torch.save(net.state_dict(),'deep_emotion-{}-{}-{}.pt'.format(epochs,batchsize,lr))
    print("===================================Training Finished===================================")
    print('true=', y_true)
    print('pred=',y_pred)
    plot_loss(train_loss_list, val_loss_list)
    plot_acc(train_acc_list,val_acc_list)
    # import pdb
    # pdb.set_trace()
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, 'confusion_matrix.png', title='confusion matrix')


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)
    classes = ['Angry', 'Disgust', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%0.2f%%" % (c*100), color='red', fontsize=15, va='center', ha='center')
        # if c > 0.001:
        #     plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()    

def plot_loss(train, val):
    host = host_subplot(111)  # row=1 col=1 first pic
    # plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    
 
    # set labels
    host.set_xlabel("epochs")
    host.set_ylabel("loss")
 
    # plot curves
    p1, = host.plot(range(len(train)), train, label="train")
    p2, = host.plot(range(len(val)), val, label="val")
 
    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)
 
    # set label color
    # host.axis["left"].label.set_color(p1.get_color())
    # par1.axis["right"].label.set_color(p2.get_color())
 
    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])
 
    plt.draw()
    plt.savefig('loss_pic.png')
    plt.show()

def plot_acc(train, val):
    host = host_subplot(111) 

    host.set_xlabel("epochs")
    host.set_ylabel("accuracy(%)")
 
    # plot curves
    p1, = host.plot(range(len(train)), train, label="train")
    p2, = host.plot(range(len(val)), val, label="val")

    host.legend(loc=5)

 
    plt.draw()
    plt.savefig('acc_pic.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-s', '--setup', type=bool, help='setup the dataset for the first time')
    parser.add_argument('-d', '--data', type=str,required= True,
                               help='data folder that contains data files that downloaded from kaggle (train.csv and test.csv)')
    parser.add_argument('-hparams', '--hyperparams', type=bool,
                               help='True when changing the hyperparameters e.g (batch size, LR, num. of epochs)')
    parser.add_argument('-e', '--epochs', type= int, help= 'number of epochs')
    parser.add_argument('-lr', '--learning_rate', type= float, help= 'value of learning rate')
    parser.add_argument('-bs', '--batch_size', type= int, help= 'training/validation batch size')
    parser.add_argument('-t', '--train', type=bool, help='True when training')
    args = parser.parse_args()

    if args.setup :
        generate_dataset = Generate_data(args.data)
        generate_dataset.split_test()
        generate_dataset.save_images()
        generate_dataset.save_images('finaltest')
        generate_dataset.save_images('val')

    if args.hyperparams:
        epochs = args.epochs
        lr = args.learning_rate
        batchsize = args.batch_size
    else :
        epochs = 100
        lr = 0.0001
        batchsize = 16

    if args.train:
        net = Deep_Emotion()
        net.to(device)
        print("Model archticture: ", net)
        traincsv_file = args.data+'/'+'train.csv'
        validationcsv_file = args.data+'/'+'val.csv'
        train_img_dir = args.data+'/'+'train/'
        validation_img_dir = args.data+'/'+'val/'

        transformation= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
        train_dataset= Plain_Dataset(csv_file=traincsv_file, img_dir = train_img_dir, datatype = 'train', transform = transformation)
        validation_dataset= Plain_Dataset(csv_file=validationcsv_file, img_dir = validation_img_dir, datatype = 'val', transform = transformation)
        train_loader= DataLoader(train_dataset,batch_size=batchsize,shuffle = True,num_workers=0)
        val_loader=   DataLoader(validation_dataset,batch_size=batchsize,shuffle = True,num_workers=0)

        criterion= nn.CrossEntropyLoss()
        optmizer= optim.Adam(net.parameters(),lr= lr,weight_decay=0.0001)
        Train(epochs, train_loader, val_loader, criterion, optmizer, device)