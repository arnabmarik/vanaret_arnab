import os
os.system('touch 1 2 3 4 5 6 7')

import dagger
dag = dagger.dagger()

 # Add nodes and others they depend on.
dag.add('3', ['4','5'])
dag.add('6', ['3','7'])
# Force this node to be old, and all its dependent parents.
dag.stale('4')
dag.run()
dag.dot('example.dot')