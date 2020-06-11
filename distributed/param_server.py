import numpy as np
import ray


@ray.remote
class ParameterServer(object):
    def __init__(self):
        self.params = []
        self.num_param_updates = 0

    def initialize(self, initial_params):
        for param in initial_params:
            self.params.append(param)

    def update_params(self, new_params):
        """Receive and synchronize new parameters"""
        self.num_param_updates = self.num_param_updates + 1
        for new_param, idx in zip(new_params, range(len(new_params))):
            self.params[idx] = new_param

    def get_params(self):
        return self.params

    def get_update_step(self):
        return self.num_param_updates
