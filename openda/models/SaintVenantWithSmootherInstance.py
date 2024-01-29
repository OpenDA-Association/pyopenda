import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
import openda.utils.py4j_utils as utils
from openda.costFunctions.JObjects import PyTime
from openda.models.SaintVenantStochModelInstance import SaintVenantStochModelInstance


class SaintVenantWithSmootherInstance(SaintVenantStochModelInstance):
    """
    Interface of the Saint-Venant Stochastic Model Instance with Smoother

    @author Aron Schouten

    """

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