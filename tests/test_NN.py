import os
import time
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import torch
from openda.algorithms.GenericEnsembleKalmanFilter import GenericEnsembleKalmanFilter
from openda.algorithms.PINN import NN
from openda.algorithms.PINN import PINN
from openda.models.SaintVenantStochModelFactory import SaintVenantModelFactory
from openda.observers.pandas_observer import PandasObserver


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    warnings.warn('CUDA is not available, so PINN will be trained using CPU instead of GPU, which can be very slow. Please consider installing CUDA.')

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def plot(epochs, losses, val_losses, losses2, val_losses2):
    plt.figure()
    plt.plot(epochs, losses, label='Training loss PINN')
    plt.plot(epochs, val_losses, '--', label='Validation loss PINN')
    plt.plot(epochs, losses2, label='Training loss NN')
    plt.plot(epochs, val_losses2, '--', label='Validation loss NN')
    plt.title('Loss of model after training')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show(block=False)

    plt.figure()
    plt.plot(epochs, losses, label='Training loss PINN')
    plt.plot(epochs, val_losses, '--', label='Validation loss PINN')
    plt.plot(epochs, losses2, label='Training loss NN')
    plt.plot(epochs, val_losses2, '--', label='Validation loss NN')
    plt.title('Loss of model after training')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show(block=False)

def setup_enkf():
    alg_config = {
        '@mainModel': None,
        '@analysisTimes': None,
        '@ensembleModel': None
    }
    model_factory = SaintVenantModelFactory()
    obs_config = {
        'store_name': None,
        'working_dir': './observations',
        'config_file': 'obs_storm_Eunice_5min.csv',
        'labels': ['0', '6', '12', '20'],
        'std': [0.5, 0.5, 0.5, 0.5]
    }
    stoch_observer = PandasObserver(config=obs_config, scriptdir=os.path.dirname(__file__))

    enkf = GenericEnsembleKalmanFilter(50, alg_config, model_factory, stoch_observer)

    return enkf

def load_data(csv):
    df = pd.read_csv(csv, delimiter=';', header=None)
    train = df.sample(frac=0.8)
    test = df.drop(train.index)
    # test = pd.read_csv(r'tests\trainig_data\noisy_training_data.csv', delimiter=';', header=None)

    # y_min = df.iloc[:,-1].min()
    # y_max = df.iloc[:,-1].max()

    x_train = torch.tensor(train.iloc[:,:-1].values, dtype=torch.float32)
    y_train = torch.tensor(train.iloc[:,-1].values, dtype=torch.float32).view(-1, 1)

    x_test = torch.tensor(test.iloc[:,:-1].values, dtype=torch.float32)
    y_test = torch.tensor(test.iloc[:,-1].values, dtype=torch.float32).view(-1, 1)
    data = [x_train, y_train, x_test, y_test]

    return data

def test():
    data = load_data(r'tests\training_data\noisy_training_data.csv')
    enkf = setup_enkf()

    layers =  [44, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    model = PINN(device, layers, enkf, data)
    model.to(device)
    # model.load_state_dict(torch.load('PINN.pth'))

    print(model)

    ## Optimization ##
    optimizer = torch.optim.Adagrad(model.parameters())
    start_time = time.time()
    epochs, losses, val_losses = model.train_model(optimizer, n_epochs=150, batch_size=32)
    elapsed = time.time() - start_time
    print(f'Training time: {elapsed:.2f}')

    # torch.save(model.state_dict(), 'PINN.pth')

    ## NN ##
    model = NN(device, layers, enkf, data)
    model.to(device)
    print(model)
    optimizer = torch.optim.Adagrad(model.parameters())
    start_time = time.time()
    _, losses2, val_losses2 = model.train_model(optimizer, n_epochs=150, batch_size=32)
    elapsed2 = time.time() - start_time
    print(f'Training time: {elapsed2:.2f}')

    plot(epochs, losses, val_losses, losses2, val_losses2)


if __name__=='__main__':
    test()
    _ = input('\n[Enter] to continue')