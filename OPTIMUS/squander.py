import timeit, os, shutil, itertools
#system("python ../../sequential-quantum-gate-decomposer/saveunitary.py ../../sequential-quantum-gate-decomposer/examples/vqe/19CNOT.qasm ./unitary.binary");
testdata = {
3: ['ham3_102.qasm', '3_17_13.qasm', 'ex-1_166.qasm', 'miller_11.qasm'],
4: ['decod24-v0_38.qasm', 'rd32-v0_66.qasm', 'decod24-v2_43.qasm', 'rd32-v1_68.qasm'],
5: ['4gt13-v1_93.qasm', 'alu-v2_33.qasm', 'alu-v2_32.qasm', 'mod5d2_64.qasm', 'hwb4_49.qasm', 'one-two-three-v0_98.qasm', 'alu-v4_36.qasm', 'one-two-three-v1_99.qasm', 'alu-v0_26.qasm', '4gt13_92.qasm', '4mod5-v0_20.qasm', 'mod10_176.qasm', '4gt5_75.qasm', 'one-two-three-v2_100.qasm', '4gt11_83.qasm', 'alu-v1_29.qasm', '4mod7-v0_94.qasm', 'alu-v3_34.qasm', '4gt10-v1_81.qasm', '4gt5_77.qasm', 'aj-e11_165.qasm', '4mod5-v1_24.qasm', 'decod24-v1_41.qasm', 'alu-v3_35.qasm', 'alu-v4_37.qasm', 'one-two-three-v0_97.qasm', '4gt13_90.qasm', '4mod5-v0_18.qasm', '4gt13_91.qasm', 'alu-v1_28.qasm', 'one-two-three-v3_101.qasm', 'alu-v2_31.qasm', '4_49_16.qasm', '4gt5_76.qasm', '4gt11_82.qasm', 'mod10_171.qasm', 'mini-alu_167.qasm', 'mod5mils_65.qasm', 'decod24-v3_45.qasm', '4mod7-v1_96.qasm', 'mod5d1_63.qasm', '4mod5-v0_19.qasm', 'alu-v0_27.qasm', '4mod5-v1_23.qasm', 'rd32_270.qasm', '4mod5-v1_22.qasm', '4gt11_84.qasm'],
6: ['4gt4-v0_78.qasm', '4gt12-v0_88.qasm', '4gt4-v0_73.qasm', 'ex1_226.qasm', 'hwb5_53.qasm', 'sf_274.qasm', '4gt4-v0_72.qasm', '4gt12-v0_86.qasm', 'xor5_254.qasm', 'mod5adder_127.qasm', 'decod24-bdd_294.qasm', 'mod8-10_177.qasm', 'decod24-enable_126.qasm', 'mod8-10_178.qasm', '4gt12-v0_87.qasm', '4gt4-v0_80.qasm', '4gt4-v1_74.qasm', '4gt12-v1_89.qasm', '4gt4-v0_79.qasm', 'sf_276.qasm', 'ex3_229.qasm', 'graycode6_47.qasm', 'alu-v2_30.qasm'],
7: ['C17_204.qasm', 'sym6_145.qasm', 'rd53_130.qasm', 'hwb6_56.qasm', 'alu-bdd_288.qasm', 'majority_239.qasm', 'rd53_135.qasm', 'ex2_227.qasm', 'ham7_104.qasm', '4mod5-bdd_287.qasm', 'rd53_131.qasm', 'rd53_133.qasm'],
8: ['rd53_251.qasm', 'f2_232.qasm', 'rd53_138.qasm', 'urf2_152.qasm', 'urf2_277.qasm', 'cm82a_208.qasm', 'hwb7_59.qasm'],
9: ['urf5_280.qasm', 'con1_216.qasm', 'hwb8_113.qasm', 'urf1_149.qasm', 'urf5_158.qasm', 'urf1_278.qasm'],
10: ['rd73_252.qasm', 'max46_240.qasm', 'sys6-v0_111.qasm', 'ising_model_10.qasm', 'hwb9_119.qasm', 'sym9_148.qasm', 'sqn_258.qasm', 'urf3_279.qasm', 'urf3_155.qasm', 'rd73_140.qasm', 'mini_alu_305.qasm', 'qft_10.qasm'],
11: ['wim_266.qasm', 'z4_268.qasm', 'dc1_220.qasm', 'urf4_187.qasm', '9symml_195.qasm', 'life_238.qasm', 'sym9_193.qasm'],
12: ['sqrt8_260.qasm', 'cycle10_2_110.qasm', 'sym9_146.qasm', 'rd84_253.qasm', 'cm152a_212.qasm', 'sym10_262.qasm'],
13: ['ising_model_13.qasm', 'dist_223.qasm', 'radd_250.qasm', 'squar5_261.qasm', 'ground_state_estimation_10.qasm', 'rd53_311.qasm', 'plus63mod4096_163.qasm', 'root_255.qasm', 'adr4_197.qasm'],
14: ['plus63mod8192_164.qasm', 'sym6_316.qasm', 'clip_206.qasm', '0410184_169.qasm', 'pm1_249.qasm', 'cm42a_207.qasm', 'sao2_257.qasm', 'cm85a_209.qasm'],
15: ['square_root_7.qasm', 'misex1_241.qasm', 'urf6_160.qasm', 'co14_215.qasm', 'dc2_222.qasm', 'rd84_142.qasm', 'ham15_107.qasm'],
16: ['cnt3-5_180.qasm', 'qft_16.qasm', 'mlp4_245.qasm', 'cnt3-5_179.qasm', 'ising_model_16.qasm', 'inc_237.qasm'] }

folder = '../../ibm_qx_mapping/examples/'
#qasm = 'con1_216' #'4gt10-v1_81' #'rd32-v1_68' #'3_17_13' #'rd53_130.qasm' #'4gt4-v0_78' #'con1_216' #'rd73_140' #'one-two-three-v2_100'
numa = 0 #1
def partitions(n, I=1):
    yield (n,)
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p
sample_methods = ["uniform"] #["uniform", "rbf", "mlp", "maxwell", "triangular", "nnc"] #only neural minimizer uses sampling methods
local_searches = ["bfgs"] #["bfgs", "psoLocal", "simplex", "gradient", "adam", "lbfgs", "gslbfgs", "grs", "random", "nelderMead", "hill"]
opt_methods = ["NeuralMinimizer", "Bfgs", "de", "ParallelDe", "Pso", "parallelPso", "DoubleGenetic", "Genetic", "IntegerGenetic", 
    "gende", "Multistart", "iPso", "ParallelGenetic", "gcrs", "Price", "Tmlsl"]
num_qbits = 3
results = {x: {} for x in testdata}
for qasm in (x.replace(".qasm", "") for x in testdata[num_qbits]):
    if os.path.exists(qasm + ".out"): os.remove(qasm + ".out")
    for opt_method in ["NeuralMinimizer"]:
        for local_search in local_searches:
            for sample_method in sample_methods:
                levels = num_qbits
                #for part in ((levelguess,),):# partitions(5):
                #    for path in {x for x in itertools.permutations(part)}:
                #    print("Path: " + str(path)) 
                cntnue = False
                #    for levels in path:
                while True:            
                    print("Running: " + opt_method + " on " + qasm + " with " + str(levels) + " levels")
                    ret = [None]
                    cmd = ("../bin/OptimusApp --filename=libsquander.so --opt_method=" +
                            opt_method + " --localsearch_method=" + local_search + " --bfgs_iterations=2001 --iterations=30" + " --neural_model=neural" + " --neural_trainmethod=lbfgs" + " --sample_method=" + sample_method + " --folder=" + folder + " --qasm=" + qasm + " --levels=" + str(levels) + ("" if not cntnue else " --continue=0"))
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
                    shutil.copyfile(qasm + ".next.gates", qasm + ".gates")
                    import numpy as np
                    from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive
                    cDecompose = qgd_N_Qubit_Decomposition_adaptive( np.eye(1<<num_qbits), level_limit_max=5, level_limit_min=0 )
                    cDecompose.set_Unitary_From_Binary(qasm + ".binary")
                    cDecompose.set_Gate_Structure_From_Binary(qasm + ".gates")
                    cDecompose.apply_Imported_Gate_Structure()
                    cost = 1.0-np.real(np.trace(cDecompose.get_Unitary())) / (1<<num_qbits)
                    if cost < 1e-8:
                        results[qasm] = (cost, t)
                        levels -= 1
                        if levels == 0: break
                    else:
                        if qasm in results: break #fail after success means we know the best level count
                        levels += 1
                    #print(cDecompose.Optimization_Problem(np.zeros(0)))
                    #cntnue = True
