# elso feladat: csak a qiskittel létrehozni az u3 kaput, és hattatni egy állapotvektoron
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

# Use Aer's qasm_simulator
simulator = QasmSimulator()

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])

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
print("\nTotal count for 00 and 11 are:",counts)



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

"""Two-pulse single-qubit gate."""
import math
from cmath import exp
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister


class U3Gate(Gate):
  
    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        label: Optional[str] = None,
    ):
        """Create new U3 gate."""
        super().__init__("u3", 1, [theta, phi, lam], label=label)

    def inverse(self):
        r"""Return inverted U3 gate.

        :math:`U3(\theta,\phi,\lambda)^{\dagger} =U3(-\theta,-\lambda,-\phi)`)
        """
        return U3Gate(-self.params[0], -self.params[2], -self.params[1])


    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Return a (multi-)controlled-U3 gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CU3Gate(*self.params, label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)


    def _define(self):
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        qc.u(self.params[0], self.params[1], self.params[2], 0)
        self.definition = qc

    def __array__(self, dtype=complex):
        """Return a Numpy.array for the U3 gate."""
        theta, phi, lam = self.params
        theta, phi, lam = float(theta), float(phi), float(lam)
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        return numpy.array(
            [
                [cos, -exp(1j * lam) * sin],
                [exp(1j * phi) * sin, exp(1j * (phi + lam)) * cos],
            ],
            dtype=dtype,
        )





 








      

