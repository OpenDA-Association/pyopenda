import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import spdiags
from scipy.stats import norm
from openda.costFunctions.JObjects import PyTime

class SaintVenantStochModelInstance:
    """
    Interface of the Saint-Venant Stochastic Model Instance

    @author Aron Schouten

    """

    def __init__(self, model_attributes, noise_config, main_or_ens=None):
        """
        Constructor
        """
        (self.param, self.param_uncertainty, self.state, self.state_uncertainty, self.sys_mean,
         self.sys_std, self.span) = model_attributes

        if noise_config is None:
            if main_or_ens == "main":
                noise_config = {'@stochParameter':False, '@stochForcing':False, '@stochInit':False}
            elif main_or_ens == "ens":
                noise_config = {'@stochParameter':False, '@stochForcing':True, '@stochInit':True}
        if noise_config.get('@stochInit'):
            realizations = [norm(loc=mean, scale=std).rvs() for mean,
                            std in zip(self.state, self.state_uncertainty)]
            self.state = realizations.copy()
        if noise_config.get('@stochParameter'):
            realizations = [norm(loc=mean, scale=std).rvs() for mean,
                            std in zip(list(self.param.values()),
                                       list(self.param_uncertainty.values()))]
            self.param = realizations

        self.auto_noise = noise_config.get('@stochForcing')

        self.current_time = PyTime(self.span[0])
        self.t = 0

    def get_time_horizon(self):
        """
        Get the computational time horizon of the model (begin and end time).

        :return: the time horizon (containing begin and end time).
        """
        return PyTime(self.span[0], self.span[2])

    def get_current_time(self):
        """
        Get the model instance's current simulation time stamp.

        :return: The model's current simulation time stamp.
        """
        return self.current_time

    def announce_observed_values(self, descriptions):
        """
        Tells model that it can expect to be asked for model values corresponding to the observations
        described. The model can make arrangement to save these values. The method compute run over a long
        interval at once, not stopping at each time with observations.
        This is meant to increase the performance especially of calibration algorithms.

        :param descriptions: an ObservationDescriptions object with meta data for the observations.
        :return:
        """
        raise NotImplementedError("Function not implemented.")

    def compute(self, time):
        """
        Let the stochastic model instance compute to the requested target time stamp.
        This function can not be used to go back in time.

        :param time: Time to compute to.
        :return:
        """
        end_time = time.get_start()
        A, B = _get_model(self.param, self.span)
        newx = self.state
        t_now = self.current_time.get_start()
        t_step = self.span[1]
        nsteps = round((end_time-t_now)/t_step)
        for _ in range(nsteps):
            self.t += self.span[1]/np.timedelta64(1,'s')
            x = self.state.copy()
            rhs = B.dot(x)
            rhs[0] = 2.5 * np.sin(2.0*np.pi/(12.*60.*60.)*self.t) #Left boundary
            newx = spsolve(A, rhs)
        self.current_time = PyTime(end_time)
        self.state = newx

    def get_observations(self, xlocs_waterlevel, xlocs_velocity=None):
        """
        Get model values corresponding to the descriptions.

        :param descriptions: An ObservationDescriptions object with meta data for the observations
        :return: python list with the model values corresponding to the descriptions
        """
        dx = self.param['L']/(self.param['n']+0.5)
        if xlocs_velocity is None:
            idx = (np.round((np.array(xlocs_waterlevel))/dx)*2).astype(int)
        else:
            idx = np.hstack((np.round((np.array(xlocs_waterlevel))/dx)*2,np.round((np.array(xlocs_velocity)-0.5*dx)/dx)*2+1)).astype(int)
        return self.state[idx]
    

    def update_state(self, state_array, main_or_ens):
        """
        Update the state vector of the model.

        :param state_array: numpy array used to update the model state.
        :param main_or_ens: "main" for updating the main model, "ens" for ensemble members.
        :return:
        """
        if main_or_ens == "ens":
            delta = state_array.copy()
            self.state += delta
        elif main_or_ens == "main":
            delta_mean = state_array.copy()
            self.state = delta_mean


    def get_state(self):
        """
        Returns the state of the model.

        :return: State vector.
        """
        return self.state


def _get_model(param, span):
    """
    Returns model matrices A and B such that A*x_new=B*x
    A and B are tri-diagonal sparce matrices, and have the order h[0], u[0], ..., h[n], u[n]  
    """
    n=param['n']
    Adata=np.zeros((3,2*n))
    Bdata=np.zeros((3,2*n))
    Adata[1,0]=1. # Left boundary
    Adata[1,2*n-1]=1. # Right boundary
    # i=1,3,5,... du/dt  + g dh/sx + f u = 0
    #  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m] 
    #= u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
    g=param['g'];f=param['f'];L=param['L']
    dt = span[1]/np.timedelta64(1,'s')
    dx = L/(n+0.5)
    temp1=0.5*g*dt/dx
    temp2=0.5*f*dt
    for i in np.arange(1,2*n-1,2):
        Adata[0,i-1]= -temp1
        Adata[1,i  ]= 1.0 + temp2
        Adata[2,i+1]= +temp1
        Bdata[0,i-1]= +temp1
        Bdata[1,i  ]= 1.0 - temp2
        Bdata[2,i+1]= -temp1
    # i=2,4,6,... dh/dt + D du/dx = 0
    #  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])  
    #= h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])
    D=param['D']
    temp1=0.5*D*dt/dx
    for i in np.arange(2,2*n,2):
        Adata[0,i-1]= -temp1
        Adata[1,i  ]= 1.0
        Adata[2,i+1]= +temp1
        Bdata[0,i-1]= +temp1
        Bdata[1,i  ]= 1.0
        Bdata[2,i+1]= -temp1    
    # Build sparse matrix
    A=spdiags(Adata,np.array([-1,0,1]),2*n,2*n).tocsr()
    B=spdiags(Bdata,np.array([-1,0,1]),2*n,2*n).tocsr()
    return A, B