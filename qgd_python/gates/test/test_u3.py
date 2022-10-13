import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, BasicAer
from qiskit import QuantumCircuit, execute, IBMQ, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit import Aer
 
#cerate unitary q-bit matrix
from scipy.stats import unitary_group

import numpy as np
backend = Aer.get_backend('unitary_simulator')

pi=np.pi

q=QuantumRegister(1,'q')
c=ClassicalRegister(1, 'c')

#Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(q, c)

#Add the u3 gate on qubit pi, pi, 
circuit.u(pi,pi, pi/2, q[0])

# Map the quantum measurement to the classical bits
#circuit.measure(q,c)

# Draw the circuit
#print(circuit)

# job execution and getting the result as an object
job = execute(circuit, backend, shots=8192)

#the result of the Qiskit job
result=job.result()

# the unitary matrix from the result object
decomposed_matrix = result.get_unitary(circuit,3)
decomposed_matrix = np.asarray(decomposed_matrix)

# Draw the circuit
print(result.get_unitary(circuit,3))



class Test_u3:
    """This is a test class of the python iterface to the u3 gate of the Python qiskit package"""

 
    def test_U3_creation(self):
        r"""
        This method is called by pytest. 
        Test to create an instance of U3 gate.

        """

        from qgd_python.gates.qgd_U3 import qgd_U3

        # number of qubits
        qbit_num = 3

        # target qbit
        target_qbit = 0

        # set the free parameters
        Theta = True
        Phi = True
        Lambda = True        

        # creating an instance of the C++ class
        U3 = qgd_U3( qbit_num, target_qbit, Theta, Phi, Lambda )
	#apply_kernel_to( u3_1qbit, input );

######



 








      

