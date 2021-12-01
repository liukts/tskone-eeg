import torch
from torch import nn
from torch._C import device
import torch.optim as optim
import numpy as np
import data_utils as dat
import model_utils as mod
import os

savedir = '211125_lstm_tskone_direct_0_05hu_lr2.00e-2/'

if not os.path.isdir("./results/"):
    os.mkdir("./results/")
if not os.path.isdir("./results/" + savedir):
    os.mkdir("./results/" + savedir)

device = 'cuda'

batch_size = 100
epochs = 15
hidden_units = 5
lr = 2.00e-2
drop_r = 0.2
layer_dim = 1
scaler_type = "minmax"
seeds = 5
model_type = 'tskone_l'

# load data
train_loader,val_loader,x_train,x_val,y_train,y_val,scaler = dat.get_data(scaler_type,test_ratio=0.2,batch_size=batch_size)

input_dim=96
output_dim=1

model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_units,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : drop_r}

loss_fn = nn.MSELoss(reduction="mean")
cc = np.empty((epochs,seeds))
rmse = np.empty_like(cc)
val_loss = np.empty_like(cc)
for i in range(0,seeds):
    print(f'Seed [{i+1}/{seeds}]')
    model = mod.get_model(model_type, model_params)
    optimizer = optim.RMSprop(model.parameters(),lr=lr)
    opt = mod.Optimizer(model=model,loss_fn=loss_fn,optimizer=optimizer,device=device)
    cc_t,rmse_t,loss_t = opt.train(train_loader,val_loader,scaler,batch_size=batch_size,
                n_epochs=epochs,n_features=input_dim,device=device)
    cc_t = np.array(cc_t)
    rmse_t = np.array(rmse_t)
    loss_t = np.array(loss_t)
    cc[:,i] = cc_t
    rmse[:,i] = rmse_t
    val_loss[:,i] = loss_t
np.save('./results/' + savedir + 'cc.npy',cc)
np.save('./results/' + savedir + 'rmse.npy',rmse)
np.save('./results/' + savedir + 'val_loss.npy',val_loss)
opt.plot_losses()