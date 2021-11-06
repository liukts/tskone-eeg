from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch

def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()

def get_data(scaler_type,test_ratio,batch_size):
    scaler = get_scaler(scaler_type)

    # load numpy arrays
    x = np.load('./processed_data/anasig_arr.npy')
    y = np.load('./processed_data/displ.npy')

    # split and transform dataset
    x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=test_ratio,shuffle=False)
    x_train_scaled = torch.Tensor(scaler.fit_transform(x_train))
    x_val_scaled = torch.Tensor(scaler.transform(x_val))
    y_train_scaled = torch.Tensor(scaler.fit_transform(y_train.reshape(-1,1)))
    y_val_scaled = torch.Tensor(scaler.transform(y_val.reshape(-1,1)))

    # put in dataloader
    train = TensorDataset(x_train_scaled,y_train_scaled)
    val = TensorDataset(x_val_scaled,y_val_scaled)
    train_loader = DataLoader(train,batch_size=batch_size,shuffle=False,drop_last=True)
    val_loader = DataLoader(val,batch_size=batch_size,shuffle=False,drop_last=True)
    return train_loader,val_loader,x_train_scaled,x_val_scaled,y_train,y_val,scaler