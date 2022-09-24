# elso feladat: csak a qiskittel létrehozni az u3 kaput, és hattatni egy állapotvektoron
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, IBMQ, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

pi=np.pi

q=QuantumRegister(1,'q')
c=ClassicalRegister(1, 'c')

# Use Aer's qasm_simulator
simulator = QasmSimulator()

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(q, c)

# Add the u3 gate on qubit pi, pi, 
circuit.u(pi,pi, pi/2, q[0])

# Map the quantum measurement to the classical bits
circuit.measure(q,c)

# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)

# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000)

# Draw the circuit
print(circuit)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(compiled_circuit)
print("\nCounts:",counts)



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

######



 








      

