def correct_all_qasm(folder):
    import glob, os
    allqbits = {}
    for filename in glob.glob(folder + "/*.qasm"):
        real_qbits = correct_qasm(filename)
        if not real_qbits in allqbits: allqbits[real_qbits] = []
        allqbits[real_qbits].append(os.path.basename(filename))
    for x in sorted(allqbits): print(x, allqbits[x])
def correct_qasm(filename):
    import re
    with open(filename, 'r') as f:
        lines = f.readlines()
        num_qbits, real_qbits = 0, 0
        for line in lines:
            if line == "OPENQASM 2.0;\n" or line == 'include "qelib1.inc";\n': continue            
            m = re.search(r"[q|c]reg [q|c]\[(\d+)\];", line)
            if not m is None: num_qbits = int(m.group(1)); continue
            m = re.search(r"(?:h|t|tdg|x|s|z) q\[(\d+)\];", line)
            if m is None: m = re.search(r"rz\([-+]?(?:\d*\.*\d+)\) q\[(\d+)\];", line)
            if not m is None: real_qbits = max(real_qbits, int(m.group(1))+1); continue
            m = re.search(r"cx q\[(\d+)\],q\[(\d+)\];", line)
            if not m is None: real_qbits = max(real_qbits, int(m.group(1))+1, int(m.group(2))+1); continue
            assert False, line
    if real_qbits != num_qbits:
        print("Removing extra qbits in " + filename + " " + str(num_qbits) + " -> " + str(real_qbits))
        lines = [re.sub(r"([q|c]reg [q|c])\[(\d+)\];", r"\1[" + str(real_qbits) + "];", line) for line in lines]
        with open(filename, 'w') as f:
            f.writelines(lines)
    return real_qbits
#correct_all_qasm("../ibm_qx_mapping/examples")

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
