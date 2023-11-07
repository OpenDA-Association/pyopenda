import numpy as np
import pandas as pd
from openda.costFunctions.JObjects import PyTime
from openda.models.SaintVenantStochModelFactory import SaintVenantModelFactory


df = pd.DataFrame()

n_iters = 50

for i in range(n_iters):
    print(i)
    model_factory = SaintVenantModelFactory()
    model = model_factory.get_instance(None, "ens")

    m = len(model.get_state()) - 1
    series_data=np.zeros((m*2 + 3))

    times = []
    next_time = model.span[0]
    while next_time < model.span[2]:
        next_time = next_time +  model.span[1]
        times.append(next_time)

    for next_time in times:
        model.compute(PyTime(next_time))

        series_data[:m]=model.get_state()[:-1] # We don't want to export the last element, as this is the AR(1) forcing component
        series_data[m:2*m]=model.get_prev_state()[:-1]
        series_data[-3:] = model.f

        df = pd.concat([df, pd.DataFrame(series_data).T])

df.to_csv(r'./tests/training_data/training_data.csv', sep=';', index=False, header=None)