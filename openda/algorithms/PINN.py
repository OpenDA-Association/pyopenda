import numpy as np
import torch
from torch import nn

class PINN(nn.Module):
    """
    Interface of the Physics-Informed Neural Network.

    Input (x_train) is a tensor containing states [state, previous_state].
    Ouptut (y_train) is a tensor containing estimated parameter(s) [f].

    @author Aron Schouten

    """
    def __init__(self, device, layers, enkf, data):
        """
        Constructor.
        """
        super().__init__()
        self.D = enkf.model_factory.model_attributes[0]['D']
        self.g = enkf.model_factory.model_attributes[0]['g']
        self.dx = enkf.model_factory.model_attributes[0]['L']/(enkf.model_factory.model_attributes[0]['n']+0.5)
        self.dt = enkf.model_factory.model_attributes[4][1]/np.timedelta64(1,'s')

        self.depth = len(layers)-2
        self.loss_function = nn.MSELoss(reduction ='mean')

        self.x_train = data[0].to(device)
        self.y_train = data[1].to(device)

        self.x_test = data[2].to(device)
        self.y_test = data[3].to(device)
              
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(self.depth+1)])
        
        self.device = device
        self.iter = 0
        
    def forward(self, x):
        """
        Forward x through the neural network.

        :param x: Observations or states.
        :return: Forwarded observations or states.
        """
        for i in range(self.depth):
            x = self.linears[i](x)
            x = self.activation(x)
        x = self.linears[-1](x)
        
        return x
                        
    def loss_param(self, x, y):
        """
        Calculates loss of estimated parameter(s).

        :param x: Observations or states.
        :param y: True value of parameter(s).
        :return: MSE of estimated parameter(s), 1/N*∑|f_est - f_real|^2.
        """
        loss = self.loss_function(self.forward(x), y)
        return loss
    
    def loss_PDE(self, x):
        """
        Calculates loss of PDE's.

        :param x: Observations or states.
        :return: MSE of PDE's, 1/N*∑|h_t + g * u_x|^2 + 1/N*∑|u_t + D * h_x + f_est * u|^2.
        """
        f = self.forward(x)

        n = x.shape[1]//2
        
        h = x[:,:n][:,0:-1:2]
        u = x[:,:n][:,1::2]
        prev_h = x[:,n:][:,0:-1:2]
        prev_u = x[:,n:][:,1::2]

        h_t = (h - prev_h)/self.dt
        u_t = (u - prev_u)/self.dt

        # Extending u and h to keep the same dimensions after taking derivative
        h_ext = torch.concat((torch.zeros(h.shape[0]).view(-1,1).to(self.device), h),-1)
        u_ext = torch.concat((torch.zeros(u.shape[0]).view(-1,1).to(self.device), u),-1)

        h_x = (h_ext[:,1:] - h_ext[:,:-1])/self.dx
        u_x = (u_ext[:,1:] - u_ext[:,:-1])/self.dx

        loss = ( (h_t + self.g*u_x)**2 ).mean()
        loss += ( (u_t + self.D*h_x + f*u)**2 ).mean()

        return loss
    
    def loss(self, x, y):
        """
        Calculates total loss.

        :param x: Observations or states.
        :param y: True value of parameter(s).
        :return: Sum of all losses.
        """
        loss_param = self.loss_param(x, y)
        loss_PDE = self.loss_PDE(x)
        # print(f'Loss for |f-f*|^2 is {loss_param}. Loss for |PDE|^2 is {loss_PDE}. Ratio = {loss_param/loss_PDE}')
        return 1000* loss_param + 500* loss_PDE

    def train_model(self, optimizer, n_epochs, batch_size):
        """
        Train the model.

        :param optimizer: The (torch.)optimizer that is used for training.
        :param n_epochs: Number of epochs.
        :param batch_size: The batch size to train with.
        :return: Lists with epoch numbers, the in-sample MSE's, out-sample MSE's, and (out-sample) biases.
        """
        output = [[], [], [], []]
        for epoch in range(n_epochs):
            for i in range(0, len(self.x_train), batch_size):
                batch_X = self.x_train[i:i+batch_size]
                batch_y = self.y_train[i:i+batch_size]

                optimizer.zero_grad()
                loss = self.loss(batch_X, batch_y)
                loss.backward()
                # print(loss)
                optimizer.step()

            mse_in, _ = self.test_model(self.x_train, self.y_train)
            mse_out, bias = self.test_model(self.x_test, self.y_test)
            output[0].append(epoch)
            output[1].append(mse_in.item())
            output[2].append(mse_out.item())
            output[3].append(bias.item())
            print(f'Epoch {epoch}: Loss = {loss}, MSE (in-sample) = {mse_in}, MSE (out-sample) = {mse_out}')

        return output
    
    def test_model(self, x, y):
        """
        Temporary puts the model in evaluation mode and calculates MSE.

        :param x: Observations or states.
        :param y: True value of parameter(s).
        :return: MSE and bias of estimated parameter(s), 1/N*∑|f_est - f_real|^2 and |f_est - f_real|.
        """
        self.eval() # Necessary when using Dropout or BatchNorm layers
        with torch.no_grad(): # To make sure we don't train when testing
            y_pred = self.forward(x)
        self.train()

        se = (y - y_pred)**2
        mse = se.mean()
        bias = y_pred.mean() - y.mean()
        return mse, bias


class NN(nn.Module):
    """
    Interface of the (ordinary) Neural Network.

    Input (x_train) is a tensor containing states [state, previous_state].
    Ouptut (y_train) is a tensor containing estimated parameter(s) [f].

    @author Aron Schouten

    """
    def __init__(self, device, layers, enkf, data):
        """
        Constructor.
        """
        super().__init__()
        self.D = enkf.model_factory.model_attributes[0]['D']
        self.g = enkf.model_factory.model_attributes[0]['g']
        self.dx = enkf.model_factory.model_attributes[0]['L']/(enkf.model_factory.model_attributes[0]['n']+0.5)
        self.dt = enkf.model_factory.model_attributes[4][1]/np.timedelta64(1,'s')

        self.depth = len(layers)-2
        self.loss_function = nn.MSELoss(reduction ='mean')

        self.x_train = data[0].to(device)
        self.y_train = data[1].to(device)

        self.x_test = data[2].to(device)
        self.y_test = data[3].to(device)
              
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(self.depth+1)])
        
        self.device = device
        self.iter = 0
        
    def forward(self, x):
        """
        Forward x through the neural network.

        :param x: Observations or states.
        :return: Forwarded observations or states.
        """
        for i in range(self.depth):
            x = self.linears[i](x)
            x = self.activation(x)
        x = self.linears[-1](x)
        
        return x
                        
    def loss(self, x, y):
        """
        Calculates loss of estimated parameter(s).

        :param x: Observations or states.
        :param y: True value of parameter(s).
        :return: MSE of estimated parameter(s), 1/N*∑|f_est - f_real|^2.
        """
        loss = self.loss_function(self.forward(x), y)
        return 1000* loss

    def train_model(self, optimizer, n_epochs, batch_size):
        """
        Train the model.

        :param optimizer: The (torch.)optimizer that is used for training.
        :param n_epochs: Number of epochs.
        :param batch_size: The batch size to train with.
        :return: Lists with epoch numbers, the in-sample MSE's, out-sample MSE's, and (out-sample) biases.
        """
        output = [[], [], [], []]
        for epoch in range(n_epochs):
            for i in range(0, len(self.x_train), batch_size):
                batch_X = self.x_train[i:i+batch_size]
                batch_y = self.y_train[i:i+batch_size]

                optimizer.zero_grad()
                loss = self.loss(batch_X, batch_y)
                loss.backward()
                # print(loss)
                optimizer.step()

            mse_in, _ = self.test_model(self.x_train, self.y_train)
            mse_out, bias = self.test_model(self.x_test, self.y_test)
            output[0].append(epoch)
            output[1].append(mse_in.item())
            output[2].append(mse_out.item())
            output[3].append(bias.item())
            print(f'Epoch {epoch}: Loss = {loss}, MSE (in-sample) = {mse_in}, MSE (out-sample) = {mse_out}')

        return output
    
    def test_model(self, x, y):
        """
        Temporary puts the model in evaluation mode and calculates MSE.

        :param x: Observations or states.
        :param y: True value of parameter(s).
        :return: MSE and bias of estimated parameter(s), 1/N*∑|f_est - f_real|^2 and |f_est - f_real|.
        """
        self.eval() # Necessary when using Dropout or BatchNorm layers
        with torch.no_grad(): # To make sure we don't train when testing
            y_pred = self.forward(x)
        self.train()

        se = (y - y_pred)**2
        mse = se.mean()
        bias = y_pred.mean() - y.mean()
        return mse, bias