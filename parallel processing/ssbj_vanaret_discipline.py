"""Define exppilict classes for the 3 disciplines"""

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
        self.nx = int(pickle.load(open("input_values.p", "rb"))[0])
        self.ny = int(pickle.load(open("input_values.p", "rb"))[1])

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

# # :run model for random value
# y_str = StructureDisc()
# if __name__ == "__main__":
#     from openmdao.core.problem import Problem
#     from openmdao.core.group import Group
#     from openmdao.core.indepvarcomp import IndepVarComp
#
#     model = Group()
#     ivc = IndepVarComp()
#     ivc.add_output('x1', np.random.random_sample(y_str.nx))  # :np.ones(y1.nx) * .2)
#     ivc.add_output('y21', np.random.random_sample(y_str.ny))
#     ivc.add_output('y31', np.random.random_sample(y_str.ny))
#     ivc.add_output('z', np.random.random_sample(y_str.nx))
#
#     model.add_subsystem('des_vars',    ivc)
#     model.add_subsystem('output_comp', y_str)
#     # model.add_subsystem('output_comp', y1)
#
#     model.connect('des_vars.x1',  'output_comp.x1')
#     model.connect('des_vars.y21', 'output_comp.y21')
#     model.connect('des_vars.y31', 'output_comp.y31')
#     model.connect('des_vars.z',   'output_comp.z')
#
#     prob = Problem(model)
#     prob.setup()
#     prob.run_model()
#     print(prob['output_comp.y1'])
#     print(prob['output_comp.y12'])
#     print(prob['output_comp.y14'])
#     print(prob['output_comp.y11'])


class AerodynamicsDisc(ExplicitComponent):

    def __init__(self):
        super(AerodynamicsDisc, self).__init__()
        self.nx = int(pickle.load(open("input_values.p", "rb"))[0])
        self.ny = int(pickle.load(open("input_values.p", "rb"))[1])

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

# # :run model for random value
# y_aer = AerodynamicsDisc()
# if __name__ == "__main__":
#     from openmdao.core.problem import Problem
#     from openmdao.core.group import Group
#     from openmdao.core.indepvarcomp import IndepVarComp
#
#     model = Group()
#     ivc = IndepVarComp()
#     ivc.add_output('x2', np.random.random_sample(y_aer.nx))  # :np.ones(y1.nx) * .2)
#     ivc.add_output('y12', np.random.random_sample(y_aer.ny))
#     ivc.add_output('y32', np.random.random_sample(y_aer.ny))
#     ivc.add_output('z', np.random.random_sample(y_aer.nx))
#
#     model.add_subsystem('des_vars',    ivc)
#     model.add_subsystem('output_comp', y_aer)
#     # model.add_subsystem('output_comp', y1)
#
#     model.connect('des_vars.x2',  'output_comp.x2')
#     model.connect('des_vars.y12', 'output_comp.y12')
#     model.connect('des_vars.y32', 'output_comp.y32')
#     model.connect('des_vars.z',   'output_comp.z')
#
#     prob = Problem(model)
#     prob.setup()
#     prob.run_model()
#     print(prob['output_comp.y2'])
#     print(prob['output_comp.y21'])
#     print(prob['output_comp.y23'])
#     print(prob['output_comp.y24'])


class PropulsionDisc(ExplicitComponent):

    def __init__(self):
        super(PropulsionDisc, self).__init__()
        self.nx = int(pickle.load(open("input_values.p", "rb"))[0])
        self.ny = int(pickle.load(open("input_values.p", "rb"))[1])
    
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

# # :run model for random value
# y_pro = PropulsionDisc()
# if __name__ == "__main__":
#     from openmdao.core.problem import Problem
#     from openmdao.core.group import Group
#     from openmdao.core.indepvarcomp import IndepVarComp
#
#     model = Group()
#     ivc = IndepVarComp()
#     ivc.add_output('x3', np.random.random_sample(y_pro.nx))  # :np.ones(y1.nx) * .2)
#     ivc.add_output('y23', np.random.random_sample(y_pro.ny))
#     ivc.add_output('z', np.random.random_sample(y_pro.nx))
#
#     model.add_subsystem('des_vars',    ivc)
#     model.add_subsystem('output_comp', y_pro)
#     # model.add_subsystem('output_comp', y1)
#
#     model.connect('des_vars.x3',  'output_comp.x3')
#     model.connect('des_vars.y23', 'output_comp.y23')
#     model.connect('des_vars.z',   'output_comp.z')
#
#     prob = Problem(model)
#     prob.setup()
#     prob.run_model()
#     print(prob['output_comp.y3'])
#     print(prob['output_comp.y31'])
#     print(prob['output_comp.y32'])
#     print(prob['output_comp.y34'])
#     print(prob['output_comp.g3'])


class PerformanceDisc(ExplicitComponent):

    def __init__(self):
        super(PerformanceDisc, self).__init__()
        self.nx = int(pickle.load(open("input_values.p", "rb"))[0])
        self.ny = int(pickle.load(open("input_values.p", "rb"))[1])

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


#
# # :run model for random value
# y_per = PerformanceDisc()
# if __name__ == "__main__":
#     from openmdao.core.problem import Problem
#     from openmdao.core.group import Group
#     from openmdao.core.indepvarcomp import IndepVarComp
#
#     model = Group()
#     ivc = IndepVarComp()
#     ivc.add_output('z', np.random.random_sample(y_per.nx))  # :np.ones(y1.nx) * .2)
#     ivc.add_output('x1', np.random.random_sample(y_per.nx))
#     ivc.add_output('y21', np.random.random_sample(y_per.ny))
#     ivc.add_output('y31', np.random.random_sample(y_per.ny))
#     ivc.add_output('x2', np.random.random_sample(y_per.nx))
#     ivc.add_output('y12', np.random.random_sample(y_per.ny))
#     ivc.add_output('y32', np.random.random_sample(y_per.nx))
#     ivc.add_output('x3', np.random.random_sample(y_per.nx))
#     ivc.add_output('y23', np.random.random_sample(y_per.ny))
#
#     model.add_subsystem('des_vars',    ivc)
#     model.add_subsystem('output_comp', y_per)
#
#     model.connect('des_vars.z',  'output_comp.z')
#     model.connect('des_vars.x1', 'output_comp.x1')
#     model.connect('des_vars.y21', 'output_comp.y21')
#     model.connect('des_vars.y31', 'output_comp.y31')
#     model.connect('des_vars.x2', 'output_comp.x2')
#     model.connect('des_vars.y12', 'output_comp.y12')
#     model.connect('des_vars.y32', 'output_comp.y32')
#     model.connect('des_vars.x3', 'output_comp.x3')
#     model.connect('des_vars.y23', 'output_comp.y23')
#
#     prob = Problem(model)
#     prob.setup()
#     prob.run_model()
#     print(prob['output_comp.range'])
