import timeit, os
#system("python ../../sequential-quantum-gate-decomposer/saveunitary.py ../../sequential-quantum-gate-decomposer/examples/vqe/19CNOT.qasm ./unitary.binary");
#5 qbit: one-two-three-v2_100, 4gt10-v1_81, one_two_three-v1_99, one_two_three-v0_98, 4mod7-v1_96, aj_e11_165, alu-v2_32
#9 qbit: con1_216
#10 qbit: rd73_140
folder = '../../ibm_qx_mapping/examples/'
qasm = '4gt10-v1_81' #'3_17_13'
levels = 5
if os.path.exists(qasm + ".out"): os.remove(qasm + ".out")
for opt_method in ["NeuralMinimizer", "Bfgs", "Pso", "Genetic", "Multistart", "iPso", "Price", "gende", "de", "Tmlsl", "gcrs",
    "IntegerGenetic", "ParallelGenetic", "DoubleGenetic", "ParallelDe", "parallelPso"]:
    ret = [None]
    def runfunc():
        ret[0] = os.system("hwloc-bind --membind node:0 --cpubind node:0 -- ../bin/OptimusApp --filename=libsquander.so --opt_method=" +
            opt_method + " --folder=" + folder + " --qasm=" + qasm + " --levels=" + str(levels) + ">>" + qasm + ".out")
    t = timeit.timeit(runfunc, number=1)
    if ret[0] != 0 and ret[0] != 34304:
        print("Error", ret[0]);
        if ret[0] != 34304: continue #34304 is bad alloc on NeuralMinimizer for large # of parameters
        break
    os.system("echo Total Time " + opt_method + ": " + str(t) + ">>" + qasm + ".out")
    os.system("tail -n 5 " + qasm + ".out") 
