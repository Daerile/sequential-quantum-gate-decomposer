from qiskit import QuantumCircuit, transpile
from qgd_python.utils import get_unitary_from_qiskit_circuit
import sys
filename = sys.argv[1]
qc_trial = QuantumCircuit.from_qasm_file( filename )
qc_trial = transpile(qc_trial, optimization_level=3, basis_gates=['cz', 'cx', 'u3'], layout_method='sabre')
Umtx_orig = get_unitary_from_qiskit_circuit( qc_trial )
from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive
cDecompose = qgd_N_Qubit_Decomposition_adaptive( Umtx_orig.conj().T, level_limit_max=5, level_limit_min=0 )
cDecompose.export_Unitary(sys.argv[2])
