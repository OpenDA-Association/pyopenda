import numpy as np
import pandas as pd
from openda.costFunctions.JObjects import PyTime
from openda.models.SaintVenantStochModelFactory import SaintVenantModelFactory


df = pd.DataFrame()

n_iters = 500_000

for i in range(n_iters):
    print(i)
    f = np.random.uniform(1e-5, 1e-3)
    model_factory = SaintVenantModelFactory(f)
    model = model_factory.get_instance(None, "main")

    # times = []
    # next_time = model.span[0]
    # while next_time < model.span[2]:
    #     next_time = next_time +  model.span[1]
    #     times.append(next_time)

    # m = len(model.get_state()) - 1
    # n = len(times)
    # series_data=np.zeros((m*2, n))

    # for i, next_time in enumerate(times):
    #     model.compute(PyTime(next_time))
    #     series_data[:m,i]=model.get_state()[:-1] # We don't want to export the last element, as this is the AR(1) forcing component
    #     series_data[m:,i]=model.get_prev_state()[:-1]

    times = []
    next_time = model.span[0]
    for _ in range(9):
        next_time = next_time +  model.span[1]
        times.append(next_time)

    for next_time in times:
        model.compute(PyTime(next_time))

    m = len(model.get_state()) - 1
    series_data=np.zeros((m*2 + 1))

    series_data[:m]=model.get_state()[:-1] # We don't want to export the last element, as this is the AR(1) forcing component
    series_data[m:-1]=model.get_prev_state()[:-1]
    series_data[-1] = f

    df = pd.concat([df, pd.DataFrame(series_data).T])

df.to_csv(r".\training_data.csv", sep=';', index=False, header=None)