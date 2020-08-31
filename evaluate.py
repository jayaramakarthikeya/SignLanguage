from torch.utils.data import Dataset
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

import onnx
import onnxruntime as ort

from dataset import get_train_test_loaders
from train import ConvNet

def evaluate(outputs : Variable , labels : Variable) -> float:
    '''Evaluate nerual network outputs with one-hotted labels vector '''
    Y = labels.numpy()
    Yhat = np.argmax(outputs , axis = 1)
    return float(np.sum(Yhat==Y))

def batch_evaluate(model : ConvNet , data_loader : torch.utils.data.DataLoader) -> float:
    score = n = 0.0
    for batch in data_loader:
        n += len(batch['image'])
        outputs = model(batch['image'])
        if isinstance(outputs , torch.Tensor):
            outputs = outputs.detach().numpy()
        score += evaluate(outputs,batch['label'][:,0])

    return (score / n)*100

def validate():
    train_loader , test_loader = get_train_test_loaders()
    model = ConvNet().float().eval()
    #Importing the pretrained weights and loading it to the existing model
    pretrained_model = torch.load('checkpoint.pth')
    model.load_state_dict(pretrained_model)

    print('Pytorch : ')
    train_acc = batch_evaluate(model,train_loader)
    print("Training Accuracy : {:.4f} ".format(train_acc))
    test_acc = batch_evaluate(model,test_loader)
    print("Test Accuracy : {:.4f} ".format(test_acc))

    train_loader , test_loader = get_train_test_loaders(batch_size=1)

    #export to onnx binary
    '''  This binary file can then be used in production to run inference with your model. 
    Most importantly, the code running this binary does not need a copy of the original network definition.
    (Disclaimer : the above sentence is copied from main documentation) '''
    fname = "signlanguage.onnx"
    dummy = torch.randn(1,1,28,28)
    torch.onnx.export(model,dummy,fname,input_names=['input'])

    #checking the exported model
    _model = onnx.load(fname)
    onnx.checker.check_model(_model)

    #create runnable session with the exported model
    ort_session = ort.InferenceSession(fname)
    model = lambda input : ort_session.run(None , {'input' : input.data.numpy()})[0]

    print('ONNX : ')
    train_acc = batch_evaluate(model,train_loader)
    print("Training Accuracy : {:.4f} ".format(train_acc))
    test_acc = batch_evaluate(model,test_loader)
    print("Test Accuracy : {:.4f} ".format(test_acc))

if __name__ == '__main__':
    validate()