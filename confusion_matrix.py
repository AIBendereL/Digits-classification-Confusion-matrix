import torch
from torch import nn


import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from torch.utils.data import DataLoader


from CNN.cnn import CNN_v0


from torchmetrics import Accuracy
from torchmetrics import ConfusionMatrix


import matplotlib.pyplot as plt
import seaborn as sbn



#Device

device = "cuda:0" if torch.cuda.is_available() else "cpu"

####



#Hyperparameter

batch_size = 64

####



#Module

def get_all_pred_labels(data_loader, model, device):
    
    num_batches = len(data_loader)
    
    all_pred = torch.empty((0), dtype=torch.int64)
    all_labels = torch.empty((0), dtype=torch.int64)
    
    all_pred = all_pred.to(device)
    all_labels = all_labels.to(device)
    
    
    for batch, (images, labels) in enumerate(data_loader):
    
        images = images.to(device)
        labels = labels.to(device)
        
        all_labels = torch.cat((all_labels, labels), dim=0)
        
        
        with torch.no_grad():
            
            pred = model(images)
            pred_class = pred.argmax(1)
            
        all_pred = torch.cat((all_pred, pred_class), dim=0)
        
        
        print(f"Current batch: {batch+1}/{num_batches}", end="\r")


    print()
    return all_pred, all_labels
            
        
def get_accuracy(all_pred, all_labels, device):
    
    metric = Accuracy(
        task = "multiclass",
        num_classes = 10
    )
    metric.to(device)
    
    
    accuracy = metric(all_pred, all_labels)
    accuracy *= 100
    
    
    return accuracy


def get_confusion_matrix(all_pred, all_labels, device):
    
    metric = ConfusionMatrix(
        task = "multiclass",
        num_classes = 10
    )
    metric.to(device)
    
    
    conf_matrix = metric(all_pred, all_labels)
    
    
    return conf_matrix


def plot_confusion_matrix(conf_matrix, class_list):
    
    conf_matrix = conf_matrix.to("cpu")
    
    
    sbn.heatmap(
        conf_matrix,
        cmap = "Blues",
        annot = True,
        fmt = "g",
        annot_kws = {
            "fontsize": "small"
        },
        cbar_kws = {
            "label": "depth"
        },
        xticklabels = class_list,
        yticklabels = class_list
    )
    
    plt.xlabel = "Predicted"
    plt.ylabel = "Actual"

    plt.show()

####



if __name__ == "__main__":
    
    #Transform
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    
    ####
    
    
    
    #Data

    train_set = datasets.MNIST(
        root = "Mnist_digits_dataset",
        train = True,
        download = True,
        transform = transform
    )
    
    val_set = datasets.MNIST(
        root = "Mnist_digits_dataset",
        train = False,
        download = True,
        transform = transform
    )
    
    
    train_loader = DataLoader(
        train_set,
        batch_size = batch_size,
        shuffle = True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size = batch_size,
        shuffle = True
    )
    
    ####
    
    
    
    #Model

    PATH = "Models/cnn_mnist_digits_50.pth"
    
    
    model = CNN_v0()
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    
    
    ####
    
    
    
    #Evaluate
    
    all_pred, all_labels = get_all_pred_labels(val_loader, model, device)

    accuracy = get_accuracy(all_pred, all_labels, device)
    
    conf_matrix = get_confusion_matrix(all_pred, all_labels, device)
    
    
    print(f"Accuracy: {accuracy:.2f}%.")
    
    
    ####
    
    
    
    #Plot
    
    class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    plot_confusion_matrix(conf_matrix, class_list)
    
    ####
    
    
    
    #Brainstorm
    

    
    ####
    
    
    
    
    
    
    
    
    