import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from openda.models.SaintVenantStochModelFactory import SaintVenantModelFactory
from openda.costFunctions.JObjects import PyTime

def plot_series(t, series_data, xlocs_waterlevel, xlocs_velocity):
    titles=[]
    ylabels=[]
    for xloc in xlocs_waterlevel:
        titles.append(f"Time Series of Waterlevel at x = {xloc*1e-3} km")
        ylabels.append("height (m)")
    for xloc in xlocs_velocity:
        titles.append(f"Time Series of Velocity at x = {xloc*1e-3} km")
        ylabels.append("velocity (m/s)")

    for i in range(series_data.shape[0]):
        fig, ax = plt.subplots(figsize=(11,7))
        ax.plot(t, series_data[i,:])
        ax.set_xlabel("time")
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))
        ax.tick_params(axis='x', labelrotation = 70)
        plt.tight_layout()
        plt.show()

def test():
    ## Initializing the model ##
    noise_config = {'@stochParameter':False, '@stochForcing':False, '@stochInit':False}
    model_factory = SaintVenantModelFactory()
    model = model_factory.get_instance(noise_config, "main")

    ## Testing the model ##
    xlocs_waterlevel = [0, 25*1e3, 50*1e3, 75*1e3, 99*1e3] # Locations (in m) where Time Series of Waterlevel is made
    xlocs_velocity = [0, 25*1e3, 50*1e3, 75*1e3] # Locations (in m) where Time Series of Velocity is made
    
    dx = model.param['L']/(model.param['n']+0.5)
    description = (np.round((np.array(xlocs_waterlevel))/dx)*2).astype(int)
    # When also using velocity use:
    # description = np.hstack((np.round((np.array(xlocs_waterlevel))/dx)*2,np.round((np.array(xlocs_velocity)-0.5*dx)/dx)*2+1)).astype(int)

    times = []
    next_time = model.span[0]
    while next_time < model.span[2]:
        times.append(next_time)
        next_time = next_time +  model.span[1]

    series_data=np.zeros((len(description),len(times)))

    for i, next_time in enumerate(times):
        model.compute(PyTime(next_time))
        series_data[:,i]=model.get_observations(description)
    
    plot_series(times, series_data, xlocs_waterlevel, xlocs_velocity)

    ## Exporting observations ##
    obs = np.vstack((list(map(str, times)), series_data)).T
    df = pd.DataFrame(obs)

    header = ["time"]
    for i in description:
        header.append(i)
    df.columns = header
    df.to_csv("obs (simulated).csv", sep=';', index=False)

if __name__ == '__main__':
    test()