import torch
from torch import nn
import torch.optim as optim
import numpy as np
import data_utils as dat
import model_utils as mod

device = 'cpu'
batch_size = 64
epochs = 20
hidden_units = 125
lr = 0.001
drop_r = 0.2
layer_dim = 2
scaler_type = "standard"

# load data
train_loader,val_loader,x_train,x_val,y_train,y_val,scaler = dat.get_data(scaler_type,test_ratio=0.2,batch_size=batch_size)

input_dim=96
output_dim=1

model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_units,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : drop_r}

model = mod.get_model('lstm', model_params)

loss_fn = nn.MSELoss(reduction="mean")
optimizer = optim.RMSprop(model.parameters(),lr=lr)
opt = mod.Optimizer(model=model,loss_fn=loss_fn,optimizer=optimizer)
opt.train(train_loader,val_loader,scaler,batch_size=batch_size,
            n_epochs=epochs,n_features=input_dim,device=device)
opt.plot_losses()