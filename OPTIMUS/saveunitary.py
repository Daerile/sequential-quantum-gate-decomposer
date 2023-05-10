def correct_all_qasm(folder):
    import glob, os
    allqbits = {}
    for filename in glob.glob(folder + "/*.qasm"):
        real_qbits = correct_qasm(filename)
        if not real_qbits in allqbits: allqbits[real_qbits] = []
        allqbits[real_qbits].append(os.path.basename(filename).replace(".qasm", ""))
    for x in sorted(allqbits): print(x, allqbits[x])
def correct_qasm(filename):
    import re
    with open(filename, 'r') as f:
        lines = f.readlines()
        num_qbits, real_qbits = 0, 0
        for line in lines:
            if line == "OPENQASM 2.0;\n" or line == 'include "qelib1.inc";\n': continue            
            m = re.search(r"qreg q\[(\d+)\];", line)
            if not m is None: num_qbits = int(m.group(1)); continue
            m = re.search(r"creg c\[(\d+)\];", line)
            if not m is None: continue
            m = re.search(r"(?:h|t|tdg|x|s|z) q\[(\d+)\];", line)
            if m is None: m = re.search(r"rz\([-+]?(?:\d*\.*\d+)\) q\[(\d+)\];", line)
            if not m is None: real_qbits = max(real_qbits, int(m.group(1))+1); continue
            m = re.search(r"cx q\[(\d+)\],q\[(\d+)\];", line)
            if not m is None: real_qbits = max(real_qbits, int(m.group(1))+1, int(m.group(2))+1); continue
            assert False, line
    if real_qbits != num_qbits or True:
        print("Removing extra qbits in " + filename + " " + str(num_qbits) + " -> " + str(real_qbits))
        lines = [re.sub(r"(qreg q)\[(\d+)\];", r"\1[" + str(real_qbits) + "];", line) for line in lines]
        lines = [line for line in lines if not "creg c" in line]
        with open(filename, 'w') as f:
            f.writelines(lines)
    return real_qbits
"""
0410184_169.qasm 14 104 1 OrderedDict([('cx', 104), ('t', 44), ('tdg', 33), ('h', 22), ('x', 8)])
3_17_13.qasm 3 22 1 OrderedDict([('cx', 17), ('t', 8), ('tdg', 6), ('h', 4), ('x', 1)])
4_49_16.qasm 5 125 1 OrderedDict([('cx', 99), ('t', 52), ('tdg', 39), ('h', 26), ('x', 1)])
4gt10-v1_81.qasm 5 84 1 OrderedDict([('cx', 66), ('t', 36), ('tdg', 27), ('h', 18), ('x', 1)])
4gt11_82.qasm 5 20 1 OrderedDict([('cx', 18), ('t', 4), ('tdg', 3), ('h', 2)])
4gt11_83.qasm 5 16 1 OrderedDict([('cx', 14), ('t', 4), ('tdg', 3), ('h', 2)])
4gt11_84.qasm 5 11 2 OrderedDict([('cx', 9), ('t', 4), ('tdg', 3), ('h', 2)])
4gt12-v0_86.qasm 6 135 1 OrderedDict([('cx', 116), ('t', 60), ('tdg', 45), ('h', 30)])
4gt12-v0_87.qasm 6 131 1 OrderedDict([('cx', 112), ('t', 60), ('tdg', 45), ('h', 30)])
4gt12-v0_88.qasm 6 108 1 OrderedDict([('cx', 86), ('t', 48), ('tdg', 36), ('h', 24)])
4gt12-v1_89.qasm 6 130 1 OrderedDict([('cx', 100), ('t', 56), ('tdg', 42), ('h', 28), ('x', 2)])
4gt13-v1_93.qasm 5 39 1 OrderedDict([('cx', 30), ('t', 16), ('tdg', 12), ('h', 8), ('x', 2)])
4gt13_90.qasm 5 65 1 OrderedDict([('cx', 53), ('t', 24), ('tdg', 18), ('h', 12)])
4gt13_91.qasm 5 61 1 OrderedDict([('cx', 49), ('t', 24), ('tdg', 18), ('h', 12)])
4gt13_92.qasm 5 38 1 OrderedDict([('cx', 30), ('t', 16), ('tdg', 12), ('h', 8)])
4gt4-v0_72.qasm 6 137 1 OrderedDict([('cx', 113), ('t', 64), ('tdg', 48), ('h', 32), ('x', 1)])
4gt4-v0_73.qasm 6 227 1 OrderedDict([('cx', 179), ('t', 96), ('tdg', 72), ('h', 48)])
4gt4-v0_78.qasm 6 137 1 OrderedDict([('cx', 109), ('t', 56), ('tdg', 42), ('h', 28)])
4gt4-v0_79.qasm 6 132 1 OrderedDict([('cx', 105), ('t', 56), ('tdg', 42), ('h', 28)])
4gt4-v0_80.qasm 6 101 1 OrderedDict([('cx', 79), ('t', 44), ('tdg', 33), ('h', 22), ('x', 1)])
4gt4-v1_74.qasm 6 154 1 OrderedDict([('cx', 119), ('t', 68), ('tdg', 51), ('h', 34), ('x', 1)])
4gt5_75.qasm 5 47 1 OrderedDict([('cx', 38), ('t', 20), ('tdg', 15), ('h', 10)])
4gt5_76.qasm 5 56 1 OrderedDict([('cx', 46), ('t', 20), ('tdg', 15), ('h', 10)])
4gt5_77.qasm 5 74 1 OrderedDict([('cx', 58), ('t', 32), ('tdg', 24), ('h', 16), ('x', 1)])
4mod5-bdd_287.qasm 7 41 1 OrderedDict([('cx', 31), ('t', 16), ('tdg', 12), ('h', 8), ('x', 3)])
4mod5-v0_18.qasm 5 40 1 OrderedDict([('cx', 31), ('t', 16), ('tdg', 12), ('h', 8), ('x', 2)])
4mod5-v0_19.qasm 5 21 1 OrderedDict([('cx', 16), ('t', 8), ('tdg', 6), ('h', 4), ('x', 1)])
4mod5-v0_20.qasm 5 12 1 OrderedDict([('cx', 10), ('t', 4), ('tdg', 3), ('h', 2), ('x', 1)])
4mod5-v1_22.qasm 5 12 1 OrderedDict([('cx', 11), ('t', 4), ('tdg', 3), ('h', 2), ('x', 1)])
4mod5-v1_23.qasm 5 41 1 OrderedDict([('cx', 32), ('t', 16), ('tdg', 12), ('h', 8), ('x', 1)])
4mod5-v1_24.qasm 5 21 1 OrderedDict([('cx', 16), ('t', 8), ('tdg', 6), ('h', 4), ('x', 2)])
4mod7-v0_94.qasm 5 92 1 OrderedDict([('cx', 72), ('t', 40), ('tdg', 30), ('h', 20)])
4mod7-v1_96.qasm 5 94 1 OrderedDict([('cx', 72), ('t', 40), ('tdg', 30), ('h', 20), ('x', 2)])
9symml_195.qasm 11 19235 1 OrderedDict([('cx', 15232), ('t', 8704), ('tdg', 6528), ('h', 4352), ('x', 65)])
C17_204.qasm 7 253 1 OrderedDict([('cx', 205), ('t', 116), ('tdg', 87), ('h', 58), ('x', 1)])
adr4_197.qasm 13 1839 1 OrderedDict([('cx', 1498), ('t', 856), ('tdg', 642), ('h', 428), ('x', 15)])
aj-e11_165.qasm 5 86 1 OrderedDict([('cx', 69), ('t', 36), ('tdg', 27), ('h', 18), ('x', 1)])
alu-bdd_288.qasm 7 48 1 OrderedDict([('cx', 38), ('t', 20), ('tdg', 15), ('h', 10), ('x', 1)])
alu-v0_26.qasm 5 49 1 OrderedDict([('cx', 38), ('t', 20), ('tdg', 15), ('h', 10), ('x', 1)])
alu-v0_27.qasm 5 21 1 OrderedDict([('cx', 17), ('t', 8), ('tdg', 6), ('h', 4), ('x', 1)])
alu-v1_28.qasm 5 22 1 OrderedDict([('cx', 18), ('t', 8), ('tdg', 6), ('h', 4), ('x', 1)])
alu-v1_29.qasm 5 22 1 OrderedDict([('cx', 17), ('t', 8), ('tdg', 6), ('h', 4), ('x', 2)])
alu-v2_30.qasm 6 285 1 OrderedDict([('cx', 223), ('t', 124), ('tdg', 93), ('h', 62), ('x', 2)])
alu-v2_31.qasm 5 255 1 OrderedDict([('cx', 198), ('t', 112), ('tdg', 84), ('h', 56), ('x', 1)])
alu-v2_32.qasm 5 92 1 OrderedDict([('cx', 72), ('t', 40), ('tdg', 30), ('h', 20), ('x', 1)])
alu-v2_33.qasm 5 22 1 OrderedDict([('cx', 17), ('t', 8), ('tdg', 6), ('h', 4), ('x', 2)])
alu-v3_34.qasm 5 30 1 OrderedDict([('cx', 24), ('t', 12), ('tdg', 9), ('h', 6), ('x', 1)])
alu-v3_35.qasm 5 22 1 OrderedDict([('cx', 18), ('t', 8), ('tdg', 6), ('h', 4), ('x', 1)])
alu-v4_36.qasm 5 66 1 OrderedDict([('cx', 51), ('t', 28), ('tdg', 21), ('h', 14), ('x', 1)])
alu-v4_37.qasm 5 22 1 OrderedDict([('cx', 18), ('t', 8), ('tdg', 6), ('h', 4), ('x', 1)])
clip_206.qasm 14 17879 1 OrderedDict([('cx', 14772), ('t', 8440), ('tdg', 6330), ('h', 4220), ('x', 65)])
cm152a_212.qasm 12 684 1 OrderedDict([('cx', 532), ('t', 304), ('tdg', 228), ('h', 152), ('x', 5)])
cm42a_207.qasm 14 940 1 OrderedDict([('cx', 771), ('t', 440), ('tdg', 330), ('h', 220), ('x', 15)])
cm82a_208.qasm 8 337 1 OrderedDict([('cx', 283), ('t', 160), ('tdg', 120), ('h', 80), ('x', 7)])
cm85a_209.qasm 14 6374 1 OrderedDict([('cx', 4986), ('t', 2848), ('tdg', 2136), ('h', 1424), ('x', 20)])
cnt3-5_179.qasm 16 61 1 OrderedDict([('cx', 85), ('t', 40), ('tdg', 30), ('h', 20)])
cnt3-5_180.qasm 16 209 1 OrderedDict([('cx', 215), ('t', 120), ('tdg', 90), ('h', 60)])
co14_215.qasm 15 8570 1 OrderedDict([('cx', 7840), ('t', 4480), ('tdg', 3360), ('h', 2240), ('x', 16)])
con1_216.qasm 9 508 1 OrderedDict([('cx', 415), ('t', 236), ('tdg', 177), ('h', 118), ('x', 8)])
cycle10_2_110.qasm 12 3386 1 OrderedDict([('cx', 2648), ('t', 1512), ('tdg', 1134), ('h', 756)])
dc1_220.qasm 11 1038 1 OrderedDict([('cx', 833), ('t', 476), ('tdg', 357), ('h', 238), ('x', 10)])
dc2_222.qasm 15 5242 1 OrderedDict([('cx', 4131), ('t', 2360), ('tdg', 1770), ('h', 1180), ('x', 21)])
decod24-bdd_294.qasm 6 40 1 OrderedDict([('cx', 32), ('t', 16), ('tdg', 12), ('h', 8), ('x', 5)])
decod24-enable_126.qasm 6 190 1 OrderedDict([('cx', 149), ('t', 84), ('tdg', 63), ('h', 42)])
decod24-v0_38.qasm 4 30 1 OrderedDict([('cx', 23), ('t', 12), ('tdg', 9), ('h', 6), ('x', 1)])
decod24-v1_41.qasm 5 50 1 OrderedDict([('cx', 38), ('t', 20), ('tdg', 15), ('h', 10), ('x', 2)])
decod24-v2_43.qasm 4 30 1 OrderedDict([('cx', 22), ('t', 12), ('tdg', 9), ('h', 6), ('x', 3)])
decod24-v3_45.qasm 5 84 1 OrderedDict([('cx', 64), ('t', 36), ('tdg', 27), ('h', 18), ('x', 5)])
dist_223.qasm 13 19694 1 OrderedDict([('cx', 16624), ('t', 9496), ('tdg', 7122), ('h', 4748), ('x', 56)])
ex-1_166.qasm 3 12 1 OrderedDict([('cx', 9), ('t', 4), ('tdg', 3), ('h', 2), ('x', 1)])
ex1_226.qasm 6 5 1 OrderedDict([('cx', 5), ('x', 2)])
ex2_227.qasm 7 355 1 OrderedDict([('cx', 275), ('t', 156), ('tdg', 117), ('h', 78), ('x', 5)])
ex3_229.qasm 6 226 1 OrderedDict([('cx', 175), ('t', 100), ('tdg', 75), ('h', 50), ('x', 3)])
f2_232.qasm 8 668 1 OrderedDict([('cx', 525), ('t', 300), ('tdg', 225), ('h', 150), ('x', 6)])
graycode6_47.qasm 6 5 1 OrderedDict([('cx', 5)])
ground_state_estimation_10.qasm 13 245614 1 OrderedDict([('cx', 154209), ('h', 123846), ('z', 41280), ('s', 41280), ('rz', 29562), ('x', 3)])
ham15_107.qasm 15 4819 1 OrderedDict([('cx', 3858), ('t', 2180), ('tdg', 1635), ('h', 1090)])
ham3_102.qasm 3 13 1 OrderedDict([('cx', 11), ('t', 4), ('tdg', 3), ('h', 2)])
ham7_104.qasm 7 185 1 OrderedDict([('cx', 149), ('t', 76), ('tdg', 57), ('h', 38)])
hwb4_49.qasm 5 134 1 OrderedDict([('cx', 107), ('t', 56), ('tdg', 42), ('h', 28)])
hwb5_53.qasm 6 758 1 OrderedDict([('cx', 598), ('t', 328), ('tdg', 246), ('h', 164)])
hwb6_56.qasm 7 3736 1 OrderedDict([('cx', 2952), ('t', 1676), ('tdg', 1257), ('h', 838)])
hwb7_59.qasm 8 13437 1 OrderedDict([('cx', 10681), ('t', 6088), ('tdg', 4566), ('h', 3044)])
hwb8_113.qasm 9 38717 1 OrderedDict([('cx', 30372), ('t', 17336), ('tdg', 13002), ('h', 8668), ('x', 2)])
hwb9_119.qasm 10 116199 1 OrderedDict([('cx', 90955), ('t', 51920), ('tdg', 38940), ('h', 25960)])
inc_237.qasm 16 5863 1 OrderedDict([('cx', 4636), ('t', 2648), ('tdg', 1986), ('h', 1324), ('x', 25)])
ising_model_10.qasm 10 70 1 OrderedDict([('rz', 280), ('h', 110), ('cx', 90)])
ising_model_13.qasm 13 71 1 OrderedDict([('rz', 370), ('h', 143), ('cx', 120)])
ising_model_16.qasm 16 71 1 OrderedDict([('rz', 460), ('h', 176), ('cx', 150)])
life_238.qasm 11 12511 1 OrderedDict([('cx', 9800), ('t', 5600), ('tdg', 4200), ('h', 2800), ('x', 45)])
majority_239.qasm 7 344 1 OrderedDict([('cx', 267), ('t', 152), ('tdg', 114), ('h', 76), ('x', 3)])
max46_240.qasm 10 14257 1 OrderedDict([('cx', 11844), ('t', 6768), ('tdg', 5076), ('h', 3384), ('x', 54)])
miller_11.qasm 3 29 1 OrderedDict([('cx', 23), ('t', 12), ('tdg', 9), ('h', 6)])
mini-alu_167.qasm 5 162 1 OrderedDict([('cx', 126), ('t', 72), ('tdg', 54), ('h', 36)])
mini_alu_305.qasm 10 69 1 OrderedDict([('cx', 77), ('t', 40), ('tdg', 30), ('h', 20), ('x', 6)])
misex1_241.qasm 15 2676 1 OrderedDict([('cx', 2100), ('t', 1200), ('tdg', 900), ('h', 600), ('x', 13)])
mlp4_245.qasm 16 10328 1 OrderedDict([('cx', 8232), ('t', 4704), ('tdg', 3528), ('h', 2352), ('x', 36)])
mod10_171.qasm 5 139 1 OrderedDict([('cx', 108), ('t', 60), ('tdg', 45), ('h', 30), ('x', 1)])
mod10_176.qasm 5 101 1 OrderedDict([('cx', 78), ('t', 44), ('tdg', 33), ('h', 22), ('x', 1)])
mod5adder_127.qasm 6 302 1 OrderedDict([('cx', 239), ('t', 136), ('tdg', 102), ('h', 68), ('x', 10)])
mod5d1_63.qasm 5 13 1 OrderedDict([('cx', 13), ('t', 4), ('tdg', 3), ('h', 2)])
mod5d2_64.qasm 5 32 1 OrderedDict([('cx', 25), ('t', 12), ('tdg', 9), ('h', 6), ('x', 1)])
mod5mils_65.qasm 5 21 1 OrderedDict([('cx', 16), ('t', 8), ('tdg', 6), ('h', 4), ('x', 1)])
mod8-10_177.qasm 6 251 1 OrderedDict([('cx', 196), ('t', 108), ('tdg', 81), ('h', 54), ('x', 1)])
mod8-10_178.qasm 6 193 1 OrderedDict([('cx', 152), ('t', 84), ('tdg', 63), ('h', 42), ('x', 1)])
one-two-three-v0_97.qasm 5 163 1 OrderedDict([('cx', 128), ('t', 72), ('tdg', 54), ('h', 36)])
one-two-three-v0_98.qasm 5 82 1 OrderedDict([('cx', 65), ('t', 36), ('tdg', 27), ('h', 18)])
one-two-three-v1_99.qasm 5 76 1 OrderedDict([('cx', 59), ('t', 32), ('tdg', 24), ('h', 16), ('x', 1)])
one-two-three-v2_100.qasm 5 40 1 OrderedDict([('cx', 32), ('t', 16), ('tdg', 12), ('h', 8), ('x', 1)])
one-two-three-v3_101.qasm 5 40 1 OrderedDict([('cx', 32), ('t', 16), ('tdg', 12), ('h', 8), ('x', 2)])
plus63mod4096_163.qasm 13 72246 1 OrderedDict([('cx', 56329), ('t', 32184), ('tdg', 24138), ('h', 16092), ('x', 1)])
plus63mod8192_164.qasm 14 105142 1 OrderedDict([('cx', 81865), ('t', 46776), ('tdg', 35082), ('h', 23388), ('x', 1)])
pm1_249.qasm 14 940 1 OrderedDict([('cx', 771), ('t', 440), ('tdg', 330), ('h', 220), ('x', 15)])
qft_10.qasm 10 63 1 OrderedDict([('rz', 90), ('cx', 90), ('h', 20)])
qft_16.qasm 16 105 1 OrderedDict([('rz', 240), ('cx', 240), ('h', 32)])
radd_250.qasm 13 1781 1 OrderedDict([('cx', 1405), ('t', 800), ('tdg', 600), ('h', 400), ('x', 8)])
rd32-v0_66.qasm 4 20 1 OrderedDict([('cx', 16), ('t', 8), ('tdg', 6), ('h', 4)])
rd32-v1_68.qasm 4 21 1 OrderedDict([('cx', 16), ('t', 8), ('tdg', 6), ('h', 4), ('x', 2)])
rd32_270.qasm 5 47 1 OrderedDict([('cx', 36), ('t', 20), ('tdg', 15), ('h', 10), ('x', 3)])
rd53_130.qasm 7 569 1 OrderedDict([('cx', 448), ('t', 256), ('tdg', 192), ('h', 128), ('x', 19)])
rd53_131.qasm 7 261 1 OrderedDict([('cx', 200), ('t', 112), ('tdg', 84), ('h', 56), ('x', 17)])
rd53_133.qasm 7 327 1 OrderedDict([('cx', 256), ('t', 144), ('tdg', 108), ('h', 72)])
rd53_135.qasm 7 159 1 OrderedDict([('cx', 134), ('t', 72), ('tdg', 54), ('h', 36)])
rd53_138.qasm 8 56 1 OrderedDict([('cx', 60), ('t', 32), ('tdg', 24), ('h', 16)])
rd53_251.qasm 8 712 1 OrderedDict([('cx', 564), ('t', 320), ('tdg', 240), ('h', 160), ('x', 7)])
rd53_311.qasm 13 124 1 OrderedDict([('cx', 124), ('t', 64), ('tdg', 48), ('h', 32), ('x', 7)])
rd73_140.qasm 10 92 1 OrderedDict([('cx', 104), ('t', 56), ('tdg', 42), ('h', 28)])
rd73_252.qasm 10 2867 1 OrderedDict([('cx', 2319), ('t', 1324), ('tdg', 993), ('h', 662), ('x', 23)])
rd84_142.qasm 15 110 1 OrderedDict([('cx', 154), ('t', 84), ('tdg', 63), ('h', 42)])
rd84_253.qasm 12 7261 1 OrderedDict([('cx', 5960), ('t', 3404), ('tdg', 2553), ('h', 1702), ('x', 39)])
root_255.qasm 13 8835 1 OrderedDict([('cx', 7493), ('t', 4280), ('tdg', 3210), ('h', 2140), ('x', 36)])
sao2_257.qasm 14 19563 1 OrderedDict([('cx', 16864), ('t', 9636), ('tdg', 7227), ('h', 4818), ('x', 32)])
sf_274.qasm 6 436 1 OrderedDict([('cx', 336), ('t', 192), ('tdg', 144), ('h', 96), ('x', 13)])
sf_276.qasm 6 435 1 OrderedDict([('cx', 336), ('t', 192), ('tdg', 144), ('h', 96), ('x', 10)])
sqn_258.qasm 10 5458 1 OrderedDict([('cx', 4459), ('t', 2548), ('tdg', 1911), ('h', 1274), ('x', 31)])
sqrt8_260.qasm 12 1659 1 OrderedDict([('cx', 1314), ('t', 748), ('tdg', 561), ('h', 374), ('x', 12)])
squar5_261.qasm 13 1049 1 OrderedDict([('cx', 869), ('t', 496), ('tdg', 372), ('h', 248), ('x', 8)])
square_root_7.qasm 15 3847 1 OrderedDict([('cx', 3089), ('tdg', 1668), ('t', 1251), ('h', 985), ('s', 417), ('x', 220)])
sym10_262.qasm 12 35572 1 OrderedDict([('cx', 28084), ('t', 16048), ('tdg', 12036), ('h', 8024), ('x', 91)])
sym6_145.qasm 7 2187 1 OrderedDict([('cx', 1701), ('t', 972), ('tdg', 729), ('h', 486)])
sym6_316.qasm 14 135 1 OrderedDict([('cx', 123), ('t', 64), ('tdg', 48), ('h', 32), ('x', 3)])
sym9_146.qasm 12 127 1 OrderedDict([('cx', 148), ('t', 80), ('tdg', 60), ('h', 40)])
sym9_148.qasm 10 12087 1 OrderedDict([('cx', 9408), ('t', 5376), ('tdg', 4032), ('h', 2688)])
sym9_193.qasm 11 19235 1 OrderedDict([('cx', 15232), ('t', 8704), ('tdg', 6528), ('h', 4352), ('x', 65)])
sys6-v0_111.qasm 10 75 1 OrderedDict([('cx', 98), ('t', 52), ('tdg', 39), ('h', 26)])
urf1_149.qasm 9 99585 1 OrderedDict([('cx', 80878), ('t', 46216), ('tdg', 34662), ('h', 23108)])
urf1_278.qasm 9 30955 1 OrderedDict([('cx', 26692), ('t', 12304), ('tdg', 9228), ('h', 6152), ('x', 390)])
urf2_152.qasm 8 44100 1 OrderedDict([('cx', 35210), ('t', 20120), ('tdg', 15090), ('h', 10060)])
urf2_277.qasm 8 11390 1 OrderedDict([('cx', 10066), ('t', 4404), ('tdg', 3303), ('h', 2202), ('x', 137)])
urf3_155.qasm 10 229365 1 OrderedDict([('cx', 185276), ('t', 105872), ('tdg', 79404), ('h', 52936)])
urf3_279.qasm 10 70702 1 OrderedDict([('cx', 60380), ('t', 28452), ('tdg', 21339), ('h', 14226), ('x', 965)])
urf4_187.qasm 11 264330 1 OrderedDict([('cx', 224028), ('t', 128016), ('tdg', 96012), ('h', 64008)])
urf5_158.qasm 9 89145 1 OrderedDict([('cx', 71932), ('t', 41104), ('tdg', 30828), ('h', 20552)])
urf5_280.qasm 9 27822 1 OrderedDict([('cx', 23764), ('t', 11440), ('tdg', 8580), ('h', 5720), ('x', 325)])
urf6_160.qasm 15 93645 1 OrderedDict([('cx', 75180), ('t', 42960), ('tdg', 32220), ('h', 21480)])
wim_266.qasm 11 514 1 OrderedDict([('cx', 427), ('t', 244), ('tdg', 183), ('h', 122), ('x', 10)])
xor5_254.qasm 6 5 1 OrderedDict([('cx', 5), ('x', 2)])
z4_268.qasm 11 1644 1 OrderedDict([('cx', 1343), ('t', 764), ('tdg', 573), ('h', 382), ('x', 11)])

3 [('3_17_13', 22), ('ex-1_166', 12), ('ham3_102', 13), ('miller_11', 29)]
4 [('decod24-v0_38', 30), ('decod24-v2_43', 30), ('rd32-v0_66', 20), ('rd32-v1_68', 21)]
5 [('4_49_16', 125), ('4gt10-v1_81', 84), ('4gt11_82', 20), ('4gt11_83', 16), ('4gt11_84', 11), ('4gt13-v1_93', 39), ('4gt13_90', 65), ('4gt13_91', 61), ('4gt13_92', 38), ('4gt5_75', 47), ('4gt5_76', 56), ('4gt5_77', 74), ('4mod5-v0_18', 40), ('4mod5-v0_19', 21), ('4mod5-v0_20', 12), ('4mod5-v1_22', 12), ('4mod5-v1_23', 41), ('4mod5-v1_24', 21), ('4mod7-v0_94', 92), ('4mod7-v1_96', 94), ('aj-e11_165', 86), ('alu-v0_26', 49), ('alu-v0_27', 21), ('alu-v1_28', 22), ('alu-v1_29', 22), ('alu-v2_31', 255), ('alu-v2_32', 92), ('alu-v2_33', 22), ('alu-v3_34', 30), ('alu-v3_35', 22), ('alu-v4_36', 66), ('alu-v4_37', 22), ('decod24-v1_41', 50), ('decod24-v3_45', 84), ('hwb4_49', 134), ('mini-alu_167', 162), ('mod10_171', 139), ('mod10_176', 101), ('mod5d1_63', 13), ('mod5d2_64', 32), ('mod5mils_65', 21), ('one-two-three-v0_97', 163), ('one-two-three-v0_98', 82), ('one-two-three-v1_99', 76), ('one-two-three-v2_100', 40), ('one-two-three-v3_101', 40), ('rd32_270', 47)]
6 [('4gt12-v0_86', 135), ('4gt12-v0_87', 131), ('4gt12-v0_88', 108), ('4gt12-v1_89', 130), ('4gt4-v0_72', 137), ('4gt4-v0_73', 227), ('4gt4-v0_78', 137), ('4gt4-v0_79', 132), ('4gt4-v0_80', 101), ('4gt4-v1_74', 154), ('alu-v2_30', 285), ('decod24-bdd_294', 40), ('decod24-enable_126', 190), ('ex1_226', 5), ('ex3_229', 226), ('graycode6_47', 5), ('hwb5_53', 758), ('mod5adder_127', 302), ('mod8-10_177', 251), ('mod8-10_178', 193), ('sf_274', 436), ('sf_276', 435), ('xor5_254', 5)]
7 [('4mod5-bdd_287', 41), ('C17_204', 253), ('alu-bdd_288', 48), ('ex2_227', 355), ('ham7_104', 185), ('hwb6_56', 3736), ('majority_239', 344), ('rd53_130', 569), ('rd53_131', 261), ('rd53_133', 327), ('rd53_135', 159), ('sym6_145', 2187)]
8 [('cm82a_208', 337), ('f2_232', 668), ('hwb7_59', 13437), ('rd53_138', 56), ('rd53_251', 712), ('urf2_152', 44100), ('urf2_277', 11390)]
9 [('con1_216', 508), ('hwb8_113', 38717), ('urf1_149', 99585), ('urf1_278', 30955), ('urf5_158', 89145), ('urf5_280', 27822)]
10 [('hwb9_119', 116199), ('ising_model_10', 70), ('max46_240', 14257), ('mini_alu_305', 69), ('qft_10', 63), ('rd73_140', 92), ('rd73_252', 2867), ('sqn_258', 5458), ('sym9_148', 12087), ('sys6-v0_111', 75), ('urf3_155', 229365), ('urf3_279', 70702)]
11 [('9symml_195', 19235), ('dc1_220', 1038), ('life_238', 12511), ('sym9_193', 19235), ('urf4_187', 264330), ('wim_266', 514), ('z4_268', 1644)]
12 [('cm152a_212', 684), ('cycle10_2_110', 3386), ('rd84_253', 7261), ('sqrt8_260', 1659), ('sym10_262', 35572), ('sym9_146', 127)]
13 [('adr4_197', 1839), ('dist_223', 19694), ('ground_state_estimation_10', 245614), ('ising_model_13', 71), ('plus63mod4096_163', 72246), ('radd_250', 1781), ('rd53_311', 124), ('root_255', 8835), ('squar5_261', 1049)]
14 [('0410184_169', 104), ('clip_206', 17879), ('cm42a_207', 940), ('cm85a_209', 6374), ('plus63mod8192_164', 105142), ('pm1_249', 940), ('sao2_257', 19563), ('sym6_316', 135)]
15 [('co14_215', 8570), ('dc2_222', 5242), ('ham15_107', 4819), ('misex1_241', 2676), ('rd84_142', 110), ('square_root_7', 3847), ('urf6_160', 93645)]
16 [('cnt3-5_179', 61), ('cnt3-5_180', 209), ('inc_237', 5863), ('ising_model_16', 71), ('mlp4_245', 10328), ('qft_16', 105)]
"""
def get_circuit_depth(filename):
    from qiskit import QuantumCircuit
    qc_trial = QuantumCircuit.from_qasm_file( filename )
    return qc_trial.num_qubits, qc_trial.depth(), qc_trial.num_connected_components(), qc_trial.count_ops()
def all_circuit_depths(folder):
    import glob, os
    allqbits = {}
    for filename in sorted(glob.glob(folder + "/*.qasm")):
        qbits, depth, ncc, co = get_circuit_depth(filename)
        print(os.path.basename(filename), qbits, depth, ncc, co)
        if not qbits in allqbits: allqbits[qbits] = []
        allqbits[qbits].append((os.path.basename(filename).replace(".qasm", ""), depth))
    for x in sorted(allqbits): print(x, allqbits[x])
def decompose_circuit(filename):
    from qiskit import QuantumCircuit
    qc_trial = QuantumCircuit.from_qasm_file( filename )    
    print(qc_trial.draw())
    qc_trial = qc_trial.decompose()
    print(qc_trial.draw())
#decompose_circuit("../ibm_qx_mapping/examples/" + "ham3_102.qasm")
def export_unitary(filename, outfile):
    from qiskit import QuantumCircuit, transpile
    from qgd_python.utils import get_unitary_from_qiskit_circuit
    qc_trial = QuantumCircuit.from_qasm_file( filename )
    qc_trial = transpile(qc_trial, optimization_level=3, basis_gates=['cz', 'cx', 'u3'], layout_method='sabre')
    Umtx_orig = get_unitary_from_qiskit_circuit( qc_trial )
    from qgd_python.decomposition.qgd_N_Qubit_Decomposition_adaptive import qgd_N_Qubit_Decomposition_adaptive
    cDecompose = qgd_N_Qubit_Decomposition_adaptive( Umtx_orig.conj().T, level_limit_max=5, level_limit_min=0 )
    cDecompose.export_Unitary(outfile)
#correct_all_qasm("../ibm_qx_mapping/examples")
all_circuit_depths("../ibm_qx_mapping/examples")
import sys
export_unitary(sys.argv[1], sys.argv[2])
