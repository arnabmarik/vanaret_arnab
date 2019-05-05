from openmdao.api import Group
import numpy as np
from openmdao.api import IndepVarComp, ExecComp, ParallelGroup
from ssbj_vanaret_discipline import StructureDisc
from ssbj_vanaret_discipline import AerodynamicsDisc
from ssbj_vanaret_discipline import PropulsionDisc
from ssbj_vanaret_discipline import PerformanceDisc


class SsbjIdf2Mda(ParallelGroup):

    """
    Analysis for IDF formulation where couplings are managed as additional constraints
    on input/output variables of related disciplines.
    """
    def __init__(self, nx_input, ny_input, y12_initial, y23_initial, y32_initial, y21_initial, y31_initial):
        super(SsbjIdf2Mda, self).__init__()
        self.nx = nx_input
        self.ny = ny_input
        self.y12 = y12_initial
        self.y23 = y23_initial
        self.y32 = y32_initial
        self.y31 = y31_initial
        self.y21 = y21_initial

    def setup(self):
        # Design variables
        self.add_subsystem('z_ini', IndepVarComp('z',   .5 * np.ones(self.nx)), promotes=['*'])
        self.add_subsystem('x1_ini', IndepVarComp('x1', .5 * np.ones(self.nx)), promotes=['*'])
        self.add_subsystem('x2_ini', IndepVarComp('x2', .5 * np.ones(self.nx)), promotes=['*'])
        self.add_subsystem('x3_ini', IndepVarComp('x3', .5 * np.ones(self.nx)), promotes=['*'])

        # Couplings
        self.add_subsystem('y31_ini', IndepVarComp('y31', self.y31), promotes=['*'])
        self.add_subsystem('y12_ini', IndepVarComp('y12', self.y12), promotes=['*'])
        self.add_subsystem('y32_ini', IndepVarComp('y32', self.y32), promotes=['*'])
        self.add_subsystem('y23_ini', IndepVarComp('y23', self.y23), promotes=['*'])
        self.add_subsystem('y21_ini', IndepVarComp('y21', self.y21), promotes=['*'])

        # Disciplines
        self.add_subsystem('Structure', StructureDisc())
        self.add_subsystem('Aerodynamics', AerodynamicsDisc())
        self.add_subsystem('Propulsion', PropulsionDisc())
        self.add_subsystem('Performance', PerformanceDisc())

        # Shared variables z
        self.connect('z', 'Structure.z')
        self.connect('z', 'Aerodynamics.z')
        self.connect('z', 'Propulsion.z')
        self.connect('z', 'Performance.z')

        # Local variables
        self.connect('x1', 'Structure.x1')
        self.connect('x2', 'Aerodynamics.x2')
        self.connect('x3', 'Propulsion.x3')
        self.connect('x1', 'Performance.x1')
        self.connect('x2', 'Performance.x2')
        self.connect('x3', 'Performance.x3')

        # Coupling variables
        self.connect('y21', 'Structure.y21')
        self.connect('y31', 'Structure.y31')
        self.connect('y32', 'Aerodynamics.y32')
        self.connect('y12', 'Aerodynamics.y12')
        self.connect('y23', 'Propulsion.y23')

        # Objective function
        self.add_subsystem('Obj', ExecComp('obj=range'), promotes=['obj'])

        # Connections
        self.connect('Performance.range', 'Obj.range')
        # self.connect('Propulsion.y34', 'Performance.y34')
        # self.connect('Aerodynamics.y24', 'Performance.y24')
        # self.connect('Structure.y14', 'Performance.y14')
        self.connect('Aerodynamics.y21', 'Performance.y21')
        self.connect('Propulsion.y31', 'Performance.y31')
        self.connect('Propulsion.y32', 'Performance.y32')
        self.connect('Structure.y12', 'Performance.y12')
        self.connect('Aerodynamics.y23', 'Performance.y23')

        # Coupling constraints
        for i in range(self.ny):
            self.add_subsystem('con_Y12' + str(i + 1),
                               ExecComp('con_y12' + str(i + 1) + '=(y12[' + str(i) + '] - y12k[' + str(i) + ']) ** 2',
                                        y12=self.y12,
                                        y12k=self.y12
                                        ),
                               promotes=['con_y12' + str(i + 1)])
            self.connect('Structure.y12', 'con_Y12' + str(i + 1) + '.y12')
            self.connect('y12', 'con_Y12' + str(i + 1) + '.y12k')

        for i in range(self.ny):
            self.add_subsystem('con_Y21' + str(i + 1),
                               ExecComp('con_y21' + str(i + 1) + '=(y21[' + str(i) + '] - y21k[' + str(i) + ']) ** 2',
                                        y21=self.y21,
                                        y21k=self.y21
                                        ),
                               promotes=['con_y21' + str(i + 1)])
            self.connect('Aerodynamics.y21', 'con_Y21' + str(i + 1) + '.y21')
            self.connect('y21', 'con_Y21' + str(i + 1) + '.y21k')

        for i in range(self.ny):
            self.add_subsystem('con_Y32' + str(i + 1),
                               ExecComp('con_y32' + str(i + 1) + '=(y32[' + str(i) + '] - y32k[' + str(i) + ']) ** 2',
                                        y32=self.y32,
                                        y32k=self.y32
                                        ),
                               promotes=['con_y32' + str(i + 1)])
            self.connect('Propulsion.y32', 'con_Y32' + str(i + 1) + '.y32')
            self.connect('y32', 'con_Y32' + str(i + 1) + '.y32k')

        for i in range(self.ny):
            self.add_subsystem('con_Y23' + str(i + 1),
                               ExecComp('con_y23' + str(i + 1) + '=(y23[' + str(i) + '] - y23k[' + str(i) + ']) ** 2',
                                        y23=self.y23,
                                        y23k=self.y23
                                        ),
                               promotes=['con_y23' + str(i + 1)])
            self.connect('Aerodynamics.y23', 'con_Y23' + str(i + 1) + '.y23')
            self.connect('y23', 'con_Y23' + str(i + 1) + '.y23k')

        for i in range(self.ny):
            self.add_subsystem('con_Y31' + str(i + 1),
                               ExecComp('con_y31' + str(i + 1) + '=(y31[' + str(i) + '] - y31k[' + str(i) + ']) ** 2',
                                        y31=self.y31,
                                        y31k=self.y31
                                        ),
                               promotes=['con_y31' + str(i + 1)])
            self.connect('Propulsion.y31', 'con_Y31' + str(i + 1) + '.y31')
            self.connect('y31', 'con_Y31' + str(i + 1) + '.y31k')

        # Local constraints
        for i in range(self.nx):
            self.add_subsystem('con_G1' + str(i + 1),
                               ExecComp('con_g1' + str(i + 1) + '=g1[' + str(i) + ']', g1=np.zeros(self.nx)),
                               promotes=['con_g1' + str(i + 1)])
            self.connect('Structure.g1', 'con_G1' + str(i + 1) + '.g1')

        for i in range(self.nx):
            self.add_subsystem('con_G2' + str(i + 1),
                               ExecComp('con_g2' + str(i + 1) + '=g2[' + str(i) + ']', g2=np.zeros(self.nx)),
                               promotes=['con_g2' + str(i + 1)])
            self.connect('Aerodynamics.g2', 'con_G2' + str(i + 1) + '.g2')

        for i in range(self.nx):
            self.add_subsystem('con_G3' + str(i + 1),
                               ExecComp('con_g3' + str(i + 1) + '=g3[' + str(i) + ']', g3=np.zeros(self.nx)),
                               promotes=['con_g3' + str(i + 1)])
            self.connect('Propulsion.g3', 'con_G3' + str(i + 1) + '.g3')
