"""Define explicit classes for the 3 disciplines"""

# Imports
from __future__ import division, print_function
import numpy as np
import pickle
# from __future__ import division, print_function
from openmdao.api import ExplicitComponent
from extrapolated_functions import Structure
from extrapolated_functions import Aerodynamics
from extrapolated_functions import Propulsion
from extrapolated_functions import Performance

from numba import jit


class StructureDisc(ExplicitComponent):

    def __init__(self):
        super(StructureDisc, self).__init__()
        with open("input_values.p", "rb") as f:
            a = pickle.load(f)
        self.nx = a[0]
        self.ny = a[1]

    @staticmethod
    def structure():
        structure = Structure()
        return structure

    @jit(nopython=False)
    def setup(self):
        # Global Design Variable z
        self.add_input('z', val=np.ones(self.nx) * 0.5)
        # Local Design Variable
        self.add_input('x1', val=np.ones(self.nx) * 0.5)
        # Coupling parameters
        self.add_input('y21', val=np.ones(self.ny) * 0.5)
        self.add_input('y31', val=np.ones(self.ny) * 0.5)
        # Coupling output
        self.add_output('y1', val=np.ones(self.ny) * 0.5)
        self.add_output('y11', val=np.ones(self.ny) * 0.5)
        self.add_output('y12', val=np.ones(self.ny) * 0.5)
        self.add_output('y14', val=np.ones(self.ny) * 0.5)
        self.add_output('g1', val=np.ones(self.nx) * 0.5)
        self.declare_partials('*', '*', method='fd')

    @jit(nopython=False)
    def compute(self, inputs, outputs):
        t1 = np.concatenate((inputs['x1'], inputs['y21'], inputs['y31'], np.random.random_sample(self.nx + 2 * self.ny),
                             np.random.random_sample(self.nx + self.ny), inputs['z']))
        x_des = t1
        output_list = ['y1', 'y11', 'y12', 'y14', 'g1']
        for i in output_list:
            outputs[i] = np.concatenate(getattr(self.structure(), i)(self.nx, self.ny, x_des), axis=0)


class AerodynamicsDisc(ExplicitComponent):

    def __init__(self):
        super(AerodynamicsDisc, self).__init__()
        with open("input_values.p", "rb") as f:
            a = pickle.load(f)
        self.nx = a[0]
        self.ny = a[1]

    @staticmethod
    def aerodynamics():
        aerodynamics = Aerodynamics()
        return aerodynamics

    @jit(nopython=False)
    def setup(self):
        # Global Design Variable z
        self.add_input('z', val=np.ones(self.nx) * 0.5)
        # Local Design Variable
        self.add_input('x2', val=np.ones(self.nx) * 0.5)
        # Coupling parameters
        self.add_input('y12', val=np.ones(self.ny) * 0.5)
        self.add_input('y32', val=np.ones(self.ny) * 0.5)
        # Coupling output
        self.add_output('y2', val=np.ones(self.ny) * 0.5)
        self.add_output('y21', val=np.ones(self.ny) * 0.5)
        self.add_output('y23', val=np.ones(self.ny) * 0.5)
        self.add_output('y24', val=np.ones(self.ny) * 0.5)
        self.add_output('g2', val=np.ones(self.nx) * 0.5)
        self.declare_partials('*', '*',  method='fd')

    @jit(nopython=False)
    def compute(self, inputs, outputs):
        t1 = np.concatenate((np.random.random_sample(self.nx + 2 * self.ny), inputs['x2'], inputs['y12'], inputs['y32'],
                             np.random.random_sample(self.nx + self.ny), inputs['z']))
        x_des = t1
        output_list = ['y2', 'y21', 'y23', 'y24', 'g2']
        for i in output_list:
            outputs[i] = np.concatenate(getattr(self.aerodynamics(), i)(self.nx, self.ny, x_des), axis=0)
        # outputs['y1'] = np.concatenate(getattr(self.structure(), 'y1')(self.nx, self.ny, self.d, x_des), axis=0)


class PropulsionDisc(ExplicitComponent):

    def __init__(self):
        super(PropulsionDisc, self).__init__()
        with open("input_values.p", "rb") as f:
            a = pickle.load(f)
        self.nx = a[0]
        self.ny = a[1]

    @staticmethod
    def propulsion():
        propulsion = Propulsion()
        return propulsion

    @jit(nopython=False)
    def setup(self):
        # Global Design Variable z
        self.add_input('z', val=np.ones(self.nx) * 0.5)
        # Local Design Variable
        self.add_input('x3', val=np.ones(self.nx) * 0.5)
        # Coupling parameters
        self.add_input('y23', val=np.ones(self.ny) * 0.5)
        # Coupling output
        self.add_output('y3', val=np.ones(self.ny) * 0.5)
        self.add_output('y31', val=np.ones(self.ny) * 0.5)
        self.add_output('y32', val=np.ones(self.ny) * 0.5)
        self.add_output('y34', val=np.ones(self.ny) * 0.5)
        self.add_output('g3', val=np.ones(self.nx) * 0.5)
        self.declare_partials('*', '*', method='fd')

    @jit(nopython=False)
    def compute(self, inputs, outputs):
        t1 = np.concatenate((np.random.random_sample(self.nx + 2 * self.ny),
                             np.random.random_sample(self.nx + 2 * self.ny),
                            inputs['x3'], inputs['y23'], inputs['z']))
        x_des = t1
        output_list = ['y3', 'y31', 'y32', 'y34', 'g3']
        for i in output_list:
            outputs[i] = np.concatenate(getattr(self.propulsion(), i)(self.nx, self.ny, x_des), axis=0)
        # outputs['y1'] = np.concatenate(getattr(self.structure(), 'y1')(self.nx, ny, self.d, x_des), axis=0)


class PerformanceDisc(ExplicitComponent):

    def __init__(self):
        super(PerformanceDisc, self).__init__()
        with open("input_values.p", "rb") as f:
            a = pickle.load(f)
        self.nx = a[0]
        self.ny = a[1]

    @staticmethod
    def performance():
        performance = Performance()
        return performance

    @jit(nopython=False)
    def setup(self):
        # Global Design Variable z
        self.add_input('z', val=np.ones(self.nx) * 0.5)
        # Coupling parameter
        self.add_input('x1', val=np.ones(self.nx) * 0.5)
        self.add_input('y21', val=np.ones(self.ny) * 0.5)
        self.add_input('y31', val=np.ones(self.ny) * 0.5)
        self.add_input('x2', val=np.ones(self.nx) * 0.5)
        self.add_input('y12', val=np.ones(self.ny) * 0.5)
        self.add_input('y32', val=np.ones(self.ny) * 0.5)
        self.add_input('x3', val=np.ones(self.nx) * 0.5)
        self.add_input('y23', val=np.ones(self.ny) * 0.5)

        # self.add_input('y14', val=np.ones(self.ny) * 0.5)
        # self.add_input('z', val=np.ones(self.ny) * 0.5)
        # self.add_input('y24', val=np.ones(self.ny) * 0.5)
        # self.add_input('y34', val=np.ones(self.ny) * 0.5)

        # Coupling output
        self.add_output('range')
        self.declare_partials('*', '*',  method='fd')

    @jit(nopython=False)
    def compute(self, inputs, outputs):
        t1 = np.concatenate((inputs['x1'], inputs['y21'], inputs['y31'], inputs['x2'], inputs['y12'], inputs['y32'],
                             inputs['x3'], inputs['y23'], inputs['z']))
        x_des = t1
        outputs['range'] = getattr(self.performance(), 'range')(self.nx, self.ny, x_des)
