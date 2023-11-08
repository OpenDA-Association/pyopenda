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
    plt.plot(epochs, losses2, label='Training loss NN')
    plt.plot(epochs, val_losses, '--', label='Validation loss PINN')
    plt.plot(epochs, val_losses2, '--', label='Validation loss NN')
    plt.title('Loss of model after training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=False)

    plt.figure()
    plt.plot(epochs, losses, label='Training loss PINN')
    plt.plot(epochs, losses2, label='Training loss NN')
    plt.plot(epochs, val_losses, '--', label='Validation loss PINN')
    plt.plot(epochs, val_losses2, '--', label='Validation loss NN')
    plt.title('Loss of model after training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show(block=False)

def plot_testing(model, model2, data):
    _, _, x_test, y_test = data
    plt.figure()
    y_pred = model.predict(x_test.to(device)).cpu()
    plt.scatter(y_test, y_pred, marker='.', label='PINN', alpha=0.2)
    y_pred = model2.predict(x_test.to(device)).cpu()
    plt.scatter(y_test, y_pred, marker='.', label='NN', alpha=0.2)
    lst = [y_test.min(), y_test.max()]
    plt.plot(lst, lst, '--k')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
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
    train = df.sample(frac=0.9)
    test = df.drop(train.index)

    x_train = torch.tensor(train.iloc[:,:-3].values, dtype=torch.float32)
    y_train = torch.tensor(train.iloc[:,-3:].values, dtype=torch.float32).view(-1, 3)

    x_test = torch.tensor(test.iloc[:,:-3].values, dtype=torch.float32)
    y_test = torch.tensor(test.iloc[:,-3:].values, dtype=torch.float32).view(-1, 3)
    data = [x_train, y_train, x_test, y_test]

    return data

def test():
    data = load_data(r'./tests/training_data/training_data_space_dep.csv')
    enkf = setup_enkf()

    layers = [44, 10, 10, 10, 10, 3]
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
    model2 = NN(device, layers, enkf, data)
    model2.to(device)
    print(model2)
    optimizer = torch.optim.Adagrad(model2.parameters())
    start_time = time.time()
    _, losses2, val_losses2 = model2.train_model(optimizer, n_epochs=150, batch_size=32)
    elapsed2 = time.time() - start_time
    print(f'Training time: {elapsed2:.2f}')

    plot_testing(model, model2, data)
    plot(epochs, losses, val_losses, losses2, val_losses2)

    assert losses[0] >= losses[-1] # PINN should have trained over time


if __name__=='__main__':
    test()
    _ = input('\n[Enter] to continue')