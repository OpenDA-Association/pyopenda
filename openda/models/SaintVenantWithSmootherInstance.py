import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
import openda.utils.py4j_utils as utils
from openda.costFunctions.JObjects import PyTime


class SaintVenantWithSmootherInstance:
    """
    Interface of the Saint-Venant Stochastic Model Instance with Smoother

    @author Aron Schouten

    """

    def __init__(self, model_attributes, noise_config, main_or_ens=None):
        """
        Constructor
        """
        (self.param, self.param_uncertainty, self.state, self.state_uncertainty, self.span) = model_attributes

        if noise_config is None:
            if main_or_ens == "main":
                noise_config = {'@stochParameter':False, '@stochForcing':False, '@stochInit':False}
            elif main_or_ens == "ens":
                noise_config = {'@stochParameter':False, '@stochForcing':True, '@stochInit':True}
            else:
                raise ValueError("main_or_ens must have value 'main' or 'ens'")
            
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
        self.state = np.array(self.state)
        self.prev_state = np.zeros_like(self.state)
        self.t = 0
        self.f = self.param['f']

    def get_time_horizon(self):
        """
        Get the computational time horizon of the model (begin and end time).

        :return: the time horizon (containing begin and end time).
        """
        return PyTime(self.span[0], None, self.span[2])

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
        return descriptions

    def compute(self, time):
        """
        Let the stochastic model instance compute to the requested target time stamp.
        This function can not be used to go back in time.

        :param time: Time to compute to.
        :return:
        """
        A, B, phi = self._get_model()
        self.prev_state = self.state.copy()
        end_time = time.get_start()
        std = np.sqrt(1-phi**2) * 0.2 # Std of model noise chosen according to desired std of AR(1)
        newx = self.state
        t_now = self.current_time.get_start()
        t_step = self.span[1]
        nsteps = round((end_time-t_now)/t_step)
        for _ in range(nsteps):
            self.t += self.span[1]/np.timedelta64(1,'s')
            x = self.state.copy()
            rhs = B.dot(x)
            rhs[0] += -0.25 + 1.25 * np.sin(2.0*np.pi/(12.42*60.*60.)*self.t) # Left boundary
            if self.auto_noise:
                rhs[-1] += norm(loc=0, scale=std).rvs() # White noise
            newx = spsolve(A, rhs)

            self.f += norm(loc=0, scale=0.0005*std).rvs() # White noise for Smoother (2000 times smaller than model noise)

        self.current_time = PyTime(end_time)
        self.state = newx

    def get_observations(self, description):
        """
        Get model values corresponding to the descriptions.

        :param descriptions: An ObservationDescriptions object with meta data for the observations
        :return: python list with the model values corresponding to the descriptions
        """
        # If necessary, first convert to integers
        if isinstance(description, np.ndarray):
            description = description.tolist()
        if isinstance(description, list):
            if isinstance(description[0], str):
                description = list(map(int,description))
        else:
            description = list(map(int,description.observation_id))
     
        return self.state[description]

    def update_state(self, state_array, main_or_ens):
        """
        Update the state vector of the model.

        :param state_array: numpy array used to update the model state.
        :param main_or_ens: "main" for updating the main model, "ens" for ensemble members.
        :return:
        """
        if main_or_ens == "ens":
            delta = utils.input_to_py_list(state_array)
            self.state += delta[:-1]
            self.f[0] += delta[-1]
        elif main_or_ens == "main":
            delta_mean = utils.input_to_py_list(state_array)
            self.state = delta_mean[:-1]
            self.f = [delta_mean[-1]]


    def get_state(self):
        """
        Returns the state of the model.

        :return: State vector.
        """
        state = np.zeros(len(self.state) + 1)
        state[:-1] = self.state
        state[-1] = self.f[0]

        return state
    
    def _get_model(self):
        """
        Returns model matrices A and B such that A*x_new=B*x
        A and B are tri-diagonal sparce matrices, and have the order h[0], u[0], ..., h[n], u[n]  
        """
        n=self.param['n']
        Adata=np.zeros((3,2*n+1))
        Bdata=np.zeros((3,2*n+1))
        Adata[1,0]=1. # Left boundary
        Adata[1,2*n-1]=1. # Right boundary
        Adata[1,2*n] = 1
        phi = np.exp( -(self.span[1]/np.timedelta64(1,'s'))/(6.*60.*60.) )
        Bdata[1,2*n] = phi
        # i=1,3,5,... du/dt  + g dh/sx + f u = 0
        #  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m] 
        #= u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
        dt = self.span[1]/np.timedelta64(1,'s')
        dx = self.param['L']/(n+0.5)
        temp1=0.5*self.param['g']*dt/dx
        for i in np.arange(1,2*n-1,2):
            temp2=0.5*self.f[int(i/(2*n)*len(self.f))]*dt
            Adata[0,i-1]= -temp1
            Adata[1,i  ]= 1.0 + temp2
            Adata[2,i+1]= +temp1
            Bdata[0,i-1]= +temp1
            Bdata[1,i  ]= 1.0 - temp2
            Bdata[2,i+1]= -temp1
        # i=2,4,6,... dh/dt + D du/dx = 0
        #  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])  
        #= h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])
        temp1=0.5*self.param['D']*dt/dx
        for i in np.arange(2,2*n,2):
            Adata[0,i-1]= -temp1
            Adata[1,i  ]= 1.0
            Adata[2,i+1]= +temp1
            Bdata[0,i-1]= +temp1
            Bdata[1,i  ]= 1.0
            Bdata[2,i+1]= -temp1    
        # Build sparse matrix
        A=spdiags(Adata,np.array([-1,0,1]),2*n+1,2*n+1).tolil()
        A[0, -1]=-1

        B=spdiags(Bdata,np.array([-1,0,1]),2*n+1,2*n+1)
        
        return A.tocsr(), B.tocsr(), phi