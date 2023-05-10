import timeit, os, shutil, itertools
#system("python ../../sequential-quantum-gate-decomposer/saveunitary.py ../../sequential-quantum-gate-decomposer/examples/vqe/19CNOT.qasm ./unitary.binary");
testdata = {
3: ['ham3_102', '3_17_13', 'ex-1_166', 'miller_11'],
4: ['decod24-v0_38', 'rd32-v0_66', 'decod24-v2_43', 'rd32-v1_68'],
5: ['4gt13-v1_93', 'alu-v2_33', 'alu-v2_32', 'mod5d2_64', 'hwb4_49', 'one-two-three-v0_98', 'alu-v4_36', 'one-two-three-v1_99', 'alu-v0_26', '4gt13_92', '4mod5-v0_20', 'mod10_176', '4gt5_75', 'one-two-three-v2_100', '4gt11_83', 'alu-v1_29', '4mod7-v0_94', 'alu-v3_34', '4gt10-v1_81', '4gt5_77', 'aj-e11_165', '4mod5-v1_24', 'decod24-v1_41', 'alu-v3_35', 'alu-v4_37', 'one-two-three-v0_97', '4gt13_90', '4mod5-v0_18', '4gt13_91', 'alu-v1_28', 'one-two-three-v3_101', 'alu-v2_31', '4_49_16', '4gt5_76', '4gt11_82', 'mod10_171', 'mini-alu_167', 'mod5mils_65', 'decod24-v3_45', '4mod7-v1_96', 'mod5d1_63', '4mod5-v0_19', 'alu-v0_27', '4mod5-v1_23', 'rd32_270', '4mod5-v1_22', '4gt11_84'],
6: ['4gt4-v0_78', '4gt12-v0_88', '4gt4-v0_73', 'ex1_226', 'hwb5_53', 'sf_274', '4gt4-v0_72', '4gt12-v0_86', 'xor5_254', 'mod5adder_127', 'decod24-bdd_294', 'mod8-10_177', 'decod24-enable_126', 'mod8-10_178', '4gt12-v0_87', '4gt4-v0_80', '4gt4-v1_74', '4gt12-v1_89', '4gt4-v0_79', 'sf_276', 'ex3_229', 'graycode6_47', 'alu-v2_30'],
7: ['C17_204', 'sym6_145', 'rd53_130', 'hwb6_56', 'alu-bdd_288', 'majority_239', 'rd53_135', 'ex2_227', 'ham7_104', '4mod5-bdd_287', 'rd53_131', 'rd53_133'],
8: ['rd53_251', 'f2_232', 'rd53_138', 'urf2_152', 'urf2_277', 'cm82a_208', 'hwb7_59'],
9: ['urf5_280', 'con1_216', 'hwb8_113', 'urf1_149', 'urf5_158', 'urf1_278'],
10: ['rd73_252', 'max46_240', 'sys6-v0_111', 'ising_model_10', 'hwb9_119', 'sym9_148', 'sqn_258', 'urf3_279', 'urf3_155', 'rd73_140', 'mini_alu_305', 'qft_10'],
11: ['wim_266', 'z4_268', 'dc1_220', 'urf4_187', '9symml_195', 'life_238', 'sym9_193'],
12: ['sqrt8_260', 'cycle10_2_110', 'sym9_146', 'rd84_253', 'cm152a_212', 'sym10_262'],
13: ['ising_model_13', 'dist_223', 'radd_250', 'squar5_261', 'ground_state_estimation_10', 'rd53_311', 'plus63mod4096_163', 'root_255', 'adr4_197'],
14: ['plus63mod8192_164', 'sym6_316', 'clip_206', '0410184_169', 'pm1_249', 'cm42a_207', 'sao2_257', 'cm85a_209'],
15: ['square_root_7', 'misex1_241', 'urf6_160', 'co14_215', 'dc2_222', 'rd84_142', 'ham15_107'],
16: ['cnt3-5_180', 'qft_16', 'mlp4_245', 'cnt3-5_179', 'ising_model_16', 'inc_237']
 }

folder = '../../ibm_qx_mapping/examples/'
#qasm = 'con1_216' #'4gt10-v1_81' #'rd32-v1_68' #'3_17_13' #'rd53_130.qasm' #'4gt4-v0_78' #'con1_216' #'rd73_140' #'one-two-three-v2_100'
numa = 1 #0
def partitions(n, I=1):
    yield (n,)
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p
sample_methods = ["uniform"] #["uniform", "rbf", "mlp", "maxwell", "triangular", "nnc"] #only neural minimizer uses sampling methods
local_searches = ["adam"] #["bfgs", "psoLocal", "simplex", "gradient", "adam", "lbfgs", "gslbfgs", "grs", "random", "nelderMead", "hill"]
opt_methods = ["NeuralMinimizer", "Bfgs", "de", "ParallelDe", "Pso", "parallelPso", "DoubleGenetic", "Genetic", "IntegerGenetic", 
    "gende", "Multistart", "iPso", "ParallelGenetic", "gcrs", "Price", "Tmlsl"]
num_qbits = 3
use_squander = False
#results = {x: {} for x in testdata}
if use_squander:
    results = {3: {'ham3_102': (3, 4.496376604379293e-10, 11.611603129887953), '3_17_13': (3, 2.4348700833343173e-10, 12.103998687118292), 'ex-1_166': (3, 7.1601424789236034e-09, 7.595035461941734), 'miller_11': (3, 4.4637382679013626e-10, 15.195476897992194)}, 4: {'decod24-v0_38': (3, 5.448974604860268e-13, 22.738686876138672), 'rd32-v0_66': (2, 2.069759919010039e-10, 10.218776091001928), 'decod24-v2_43': (4, 1.0951350937205007e-11, 68.7201279271394), 'rd32-v1_68': (2, 3.2307490016592055e-14, 10.963908253004774)}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}, 11: {}, 12: {}, 13: {}, 14: {}, 15: {}, 16: {}}
else: #NeuralMinimizer
    results = {3: {'ham3_102': (2, 5.487906906687101e-09, 26.668307663174346), '3_17_13': (3, 8.748746505027327e-09, 1.6907278751023114), 'ex-1_166': (3, 7.520237099711835e-09, 1.2516490330453962), 'miller_11': (3, 4.50146386832273e-09, 14.192241134122014)}, 4: {'decod24-v0_38': (2, 7.084205444485292e-09, 84.25802736007608), 'rd32-v0_66': (2, 4.143922094357322e-09, 10.16023734700866), 'decod24-v2_43': (2, 7.303407212333468e-09, 29.003465383080766), 'rd32-v1_68': (2, 5.156767901581816e-09, 11.690480364020914)}, 5: {'4gt13-v1_93': (3, 8.716013466525396e-09, 4.623484890908003), 'alu-v2_33': (3, 2.1432007279997833e-09, 7.497361100045964), 'alu-v2_32': (4, 9.513836829455613e-09, 252.66068387893029), 'mod5d2_64': (2, 9.064051065266199e-09, 20.493360723135993), 'hwb4_49': (4, 7.2158514718978495e-09, 121.33502761390992)}, 6: {}, 7: {}, 8: {}, 9: {}, 10: {}, 11: {}, 12: {}, 13: {}, 14: {}, 15: {}, 16: {}}
for num_qbits in (5,6,7,8):
    for qasm in testdata[num_qbits]:
        if qasm in results[num_qbits]: continue
        #if qasm == 'one-two-three-v0_98': continue 
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
                        import numpy as np
                        from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive
                        print("Running: " + opt_method + " on " + qasm + " with " + str(levels) + " levels")
                        if use_squander:
                            cDecompose = qgd_N_Qubit_Decomposition_adaptive( np.eye(1<<num_qbits), level_limit_max=levels, level_limit_min=levels )
                            cDecompose.set_Unitary_From_Binary(qasm + ".binary")
                            t = timeit.timeit(cDecompose.Start_Decomposition, number=1)
                            cDecompose.apply_Imported_Gate_Structure()
                        else:
                            ret = [None]
                            cmd = ("../bin/OptimusApp --filename=libsquander.so --opt_method=" +
                                    opt_method + " --localsearch_method=" + local_search + " --bfgs_iterations=2001 --adam_b1=0.68 --adam_b2=0.8 --adam_rate=0.001 --adam_iterations=100 --iterations=30" + " --neural_model=neural" + " --neural_trainmethod=lbfgs" + " --sample_method=" + sample_method + " --folder=" + folder + " --qasm=" + qasm + " --levels=" + str(levels) + ("" if not cntnue else " --continue=0"))
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
                            cDecompose = qgd_N_Qubit_Decomposition_adaptive( np.eye(1<<num_qbits), level_limit_max=5, level_limit_min=0 )
                            cDecompose.set_Unitary_From_Binary(qasm + ".binary")
                            cDecompose.set_Gate_Structure_From_Binary(qasm + ".gates")
                            cDecompose.apply_Imported_Gate_Structure()
                        cost = 1.0-np.real(np.trace(cDecompose.get_Unitary())) / (1<<num_qbits)
                        if cost < 1e-8:
                            results[num_qbits][qasm] = (levels, cost, t)
                            levels -= 1
                            if levels == 0: break
                        else:
                            if qasm in results[num_qbits]: break #fail after success means we know the best level count
                            levels += 1
                        #print(cDecompose.Optimization_Problem(np.zeros(0)))
                        #cntnue = True
        print(results)
