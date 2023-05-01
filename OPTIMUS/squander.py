import timeit, os
#system("python ../../sequential-quantum-gate-decomposer/saveunitary.py ../../sequential-quantum-gate-decomposer/examples/vqe/19CNOT.qasm ./unitary.binary");
#5 qbit: one-two-three-v2_100, 4gt10-v1_81, one_two_three-v1_99, one_two_three-v0_98, 4mod7-v1_96, aj_e11_165, alu-v2_32
#9 qbit: con1_216
#10 qbit: rd73_140
folder = '../../ibm_qx_mapping/examples/'
qasm = '4gt10-v1_81' #'rd73_140' #'con1_216' #'one-two-three-v2_100' #'3_17_13'
numa = 0 #1
levels = 1 #2
if os.path.exists(qasm + ".out"): os.remove(qasm + ".out")
for opt_method in ["Genetic"]: #["NeuralMinimizer", "Bfgs", "de", "ParallelDe", "Pso", "parallelPso", "DoubleGenetic", "Genetic", "IntegerGenetic", 
    #"gende", "Multistart", "iPso", "ParallelGenetic", "gcrs", "Price", "Tmlsl"]:
    print("Running: " + opt_method + " on " + qasm)
    cntnue = True
    ret = [None]
    cmd = ("../bin/OptimusApp --filename=libsquander.so --opt_method=" +
            opt_method + " --folder=" + folder + " --qasm=" + qasm + " --levels=" + str(levels) + ("" if not cntnue else " --continue=0"))
    print(cmd)
    def runfunc():
        ret[0] = os.system("hwloc-bind --membind node:" + str(numa) + " --cpubind node:" + str(numa) + " -- " + cmd + ">>" + qasm + ".out")
    t = timeit.timeit(runfunc, number=1)
    if ret[0] != 0:
        print("Error", ret[0]);
        if ret[0] == 34304: continue #34304 is bad alloc on NeuralMinimizer for large # of parameters
        break
    os.system("echo Total Time " + opt_method + ": " + str(t) + ">>" + qasm + ".out")
    os.system("tail -n 5 " + qasm + ".out") 
