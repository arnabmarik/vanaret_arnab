
# Imports
from openmdao.api import Group, ParallelGroup
import numpy as np
from openmdao.api import IndepVarComp, ExecComp
from openmdao.solvers.linear.linear_block_gs import LinearBlockGS

from ssbj_vanaret_discipline import StructureDisc
from ssbj_vanaret_discipline import AerodynamicsDisc
from ssbj_vanaret_discipline import PropulsionDisc
from ssbj_vanaret_discipline import PerformanceDisc
from openmdao.api import NonlinearBlockGS, ScipyKrylov, NonlinearBlockJac
import numba
from numba import jit, vectorize


class SsbjMda(Group):
    """
    SSBJ Analysis with aerodynamics, common, propulsion and structure disciplines.
    """
    def __init__(self, nx_input):
        super(SsbjMda, self).__init__()
        # self.scalers = scalers
        self.nx = nx_input

    def setup(self):
        # Design variables
        self.add_subsystem('z_ini', IndepVarComp('z',   .5 * np.ones(self.nx)), promotes=['*'])
        self.add_subsystem('x1_ini', IndepVarComp('x1', .5 * np.ones(self.nx)), promotes=['*'])
        self.add_subsystem('x2_ini', IndepVarComp('x2', .5 * np.ones(self.nx)), promotes=['*'])
        self.add_subsystem('x3_ini', IndepVarComp('x3', .5 * np.ones(self.nx)), promotes=['*'])

        # Discipline
        sap_group = Group()
        sap_group.add_subsystem('Structure', StructureDisc(), promotes=['*'])
        sap_group.add_subsystem('Aerodynamics', AerodynamicsDisc(), promotes=['*'])
        sap_group.add_subsystem('Propulsion', PropulsionDisc(), promotes=['*'])

        sap_group.nonlinear_solver = NonlinearBlockGS()
        # sap_group.nonlinear_solver.options['atol'] = 1.0e-3
        sap_group.nonlinear_solver.options['maxiter'] = 100
        sap_group.linear_solver = ScipyKrylov()
        self.add_subsystem('Mda', sap_group, promotes=['*'])

        self.add_subsystem('Performance', PerformanceDisc(), promotes=['*'])

        # Constraints
        constraint1 = []
        for i in range(self.nx):
            constraint1.append('con_g1' + str(i + 1) + ' = g1[' + str(i) + ']')
        self.add_subsystem('Constraints_g1', ExecComp(constraint1, g1=np.zeros(self.nx)), promotes=['*'])

        constraint2 = []
        for i in range(self.nx):
            constraint2.append('con_g2' + str(i + 1) + ' = g2[' + str(i) + ']')
        self.add_subsystem('Constraints_g2', ExecComp(constraint2, g2=np.zeros(self.nx)), promotes=['*'])

        constraint3 = []
        for i in range(self.nx):
            constraint3.append('con_g3' + str(i + 1) + ' = g3[' + str(i) + ']')
        self.add_subsystem('Constraints_g3', ExecComp(constraint3, g3=np.zeros(self.nx)), promotes=['*'])
