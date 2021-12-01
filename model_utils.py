import torch
from torch import nn
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from torch import Tensor

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, device):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, device):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

class TskoneLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(TskoneLSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.fcout = nn.Linear(50,output_dim)

        self.t = TSKONE()

    def forward(self, x, spikes, device):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        x = self.t(x,torch.zeros_like(x).to(device))
        # x = self.t(x,spikes)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        # out = self.fcout(out)

        return out

def get_model(model, model_params):
    models = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "tskone_l": TskoneLSTMModel,
    }
    return models.get(model.lower())(**model_params)

class Optimizer:
    def __init__(self, model, loss_fn, optimizer, device):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        
    def train_step(self, x, y, spikes, device):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x,spikes,device)

        # Computes loss
        loss = torch.sqrt(self.loss_fn(y, yhat))

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, scaler, batch_size=64, n_epochs=50, n_features=1, device='cuda'):
        model_path = f'models/{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        cc = []
        rmse = []
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch, spikes in tqdm(train_loader, desc='train', unit='batch', ncols=80, leave=False):
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                spikes = spikes.to(device)
                loss = self.train_step(x_batch, y_batch, spikes, device)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                predictions = []
                values = []
                batch_val_losses = []
                for x_val, y_val, spikes in tqdm(val_loader, desc='validation', unit='batch', ncols=80, leave=False):
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    spikes = spikes.to(device)
                    self.model.eval()
                    yhat = self.model(x_val,spikes,device)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                    predictions.append(yhat.cpu().detach().numpy())
                    values.append(y_val.cpu().detach().numpy())
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
            pred_act,val_act = self.inverse_transform(scaler,np.array(predictions).reshape(-1,1),np.array(values).reshape(-1,1))
            pred_act = pred_act.reshape(-1,)
            val_act = val_act.reshape(-1,)
            print(
                f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
            )
            print(f"\tRMSE: {self.rmse(pred_act,val_act):.4f}\t CC: {self.cc(pred_act,val_act)[0]:.4f}")
            cc.append(self.cc(pred_act,val_act)[0])
            rmse.append(self.rmse(pred_act,val_act))
        torch.save(self.model.state_dict(), model_path)
        return cc,rmse,self.val_losses
 
    def evaluate(self, test_loader, batch_size=1, n_features=1, device='cuda'):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in tqdm(test_loader, desc='test', unit='batch', ncols=80, leave=False):
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.savefig('training.png')
        plt.close()

    def inverse_transform(self,scaler,pred,val):
        pred_act = scaler.inverse_transform(pred)
        val_act = scaler.inverse_transform(val)
        return pred_act,val_act

    def rmse(self,pred,val):
        rmse = mean_squared_error(val,pred)**0.5
        return rmse
    def r2(self,pred,val):
        r2 = r2_score(val,pred)
        return r2
    def cc(self,pred,val):
        cc = pearsonr(val,pred)
        return cc

class TSKONE(nn.Module):

    def __init__(self):
        super(TSKONE, self).__init__()

    def forward(self, input: Tensor, config: Tensor) -> Tensor:
        return tskone(input, config)

def tskone(input: Tensor, config: Tensor) -> Tensor:
    
    # tskone calculation, input in mT, output in dM
    return torch.where(config<0.5,-0.03828571*input**2+0.44128571*input+0.003-1.1,
                         -0.04857143*input**2+0.53471429*input-0.42914286-1.1)