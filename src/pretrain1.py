#Congming Liao 260790998
import os
import json
import sys
sys.path.insert(0, os.path.abspath("..\FaceAttributeDetection\model"))


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from tqdm import tqdm
from MobileNetV2 import MobileNetV2

def main():
    # Define hyperparameters 
    lr = 0.001
    lr_decay = 0.98
    momentum = 0.9
    weight_decay = 0.00004
    batch_size = 48

    torch.cuda.empty_cache()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: " + str(device))

    #Prepare the Dataset and Data_loader
    transform = transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
         #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
         #transforms.Normalize(mean=[-1,-1,-1],std=[2,2,2])])

    train_data = getCeleba_train(transform)
    valid_data = getCeleba_valid(transform)
    print(len(valid_data))
    #add samplers
    # indices = list(range(202599))
    # train_data_size = len(train_data)
    # valid_data_size = len(valid_data)
    # train_sampler = SubsetRandomSampler(indices[:train_data_size])
    # valid_sampler = SubsetRandomSampler(indices[train_data_size:valid_data_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size)#,sampler = train_sampler)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)#,sampler = valid_sampler)
    val_num = len(valid_loader)
    train_num = len(train_loader)
    

    loss_f = nn.BCELoss()
    net = MobileNetV2(num_classes = 40)
    params = [p for p in net.parameters() if p.requires_grad]
    #optimizer = optim.RMSprop(params, lr=lr, alpha = lr_decay, weight_decay = weight_decay, momentum = momentum)
    #optimizer = optim.Adam(params, lr = lr, weight_decay = weight_decay )
    optimizer = optim.SGD(params, lr = lr, momentum= momentum, weight_decay= weight_decay)
    
    model_weight_path = "src/mobilenet_v2.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location=device)

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    net.to(device)

    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False

    epochs = 10
    best_acc = 0.0
    save_path = 'data/MobileNetV2.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        correct = 0
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            labels = labels.type(torch.FloatTensor)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = net(images)
            logits = logits.to(device)
            loss = loss_f(logits, labels)

            loss.backward()
            optimizer.step()

            result = logits > 0.5
            correct += (result == labels).sum().item() 
            # print statistics
            running_loss += loss.item()
            
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(valid_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_labels = val_labels.type(torch.FloatTensor)
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                outputs = net(val_images)
                outputs = outputs.to(device)
                #loss = loss_f(outputs, val_labels)
                predict_y = outputs > 0.5
                acc += torch.eq(predict_y, val_labels).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / (val_num * 40)
        train_accurate = correct / (train_num * 40)
        print('[epoch %d] train_loss: %.3f train_accuracy %.3f validation_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, train_accurate, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')



def getCeleba_train(transform):
    # you can use download = True to download the dataset
    download = True

    train_data = datasets.CelebA(root = 'data', split = 'train', target_type = 'attr' ,                       
                                 transform = transform, download = download,)
    print("---------------train_data get!---------------")
    
    return train_data 

def getCeleba_valid(transform):
    # you can use download = True to download the dataset
    download = True

    valid_data = datasets.CelebA(root = 'data', split = 'valid', target_type = 'attr' ,                       
                                 transform = transform, download = download,)
    print("---------------valid_data get!---------------")

    return valid_data

def getCeleba_test(transform):
    # you can use download = True to download the dataset
    download = True

    test_data  = datasets.CelebA(root = 'data', split = 'test',  target_type = 'attr' ,                       
                                 transform = transform, download = download,)
    print("---------------test_data get!---------------")
    
    return test_data



if __name__ == '__main__':
    main()
