OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
u(6.5380044e-13,0.037803919,0.35803619) q[4];
u(pi/2,0.23456952,-pi) q[3];
cz q[4],q[3];
u(pi/2,-0.4083825,2.9070231) q[3];
u(pi/2,-4.4212975,pi) q[0];
cz q[3],q[0];
u(pi/2,-5.1792934,-pi/2) q[2];
u(pi,0.50121295,0.50121279) q[0];
cz q[2],q[0];
u(-pi,0.38900027,-2.2037076) q[2];
u(-1.5046278e-08,-0.0054172155,-1.6245457) q[1];
cz q[2],q[1];
u(-pi,4.3642147,-1.9450087) q[4];
u(pi/2,-4.1542205,4.2180821) q[0];
cz q[4],q[0];
u(5.0467061e-13,0.03150344,-2.351589) q[4];
u(0.91388887,-3.1316689,1.6137134) q[1];
cx q[4],q[1];
u(-3.9271711,0.022934873,0.019568461) q[1];
cx q[4],q[1];
u(pi/2,1.0393085,1.6527767) q[2];
u(1.4423709,-2.3583141,-3.1409804) q[1];
cz q[2],q[1];
u(pi/2,-2.0273056,-4.1809012) q[2];
u(pi/2,-0.5767998,-2.1289648) q[0];
cz q[2],q[0];
u(pi/2,-1.9127334,0.78273782) q[1];
u(0.71477588,-1.1241413,3.161038) q[0];
cx q[1],q[0];
u(2.0940327,0.027335959,0.023258215) q[0];
cx q[1],q[0];
u(pi,-2.2718218,-3.209344) q[3];
u(1.5830362,-pi/2,2.2217098) q[1];
cx q[3],q[1];
u(0.78533611,0.0050706853,0.005068892) q[1];
cx q[3],q[1];
u(-pi,2.1942123,1.5082505) q[4];
u(2.3399338,-1.4524393,-5.826676) q[2];
cz q[4],q[2];
u(pi,6.0832428,-2.2539596) q[4];
u(0.70841197,1.0763401,-2.0210021) q[0];
cz q[4],q[0];
u(-pi,5.0937142,2.328395) q[3];
u(pi/2,-3.4507736,1.8418973) q[0];
cz q[3],q[0];
u(1.5585567,4.2074027,-3*pi/2) q[1];
u(2.3571595,0.97192159,3.4741585) q[0];
cx q[1],q[0];
u(-1.0471847,-0.0077267149,0.0004930597) q[0];
cx q[1],q[0];
u(pi/2,-5.3225832,3.6780268) q[4];
u(pi/2,-2.3331089,6.1648283) q[2];
cz q[4],q[2];
u(-5.5090756,-0.013281884,0.96473391) q[0];
u(pi/2,pi,3.1701571) q[1];
u(pi,0.046171121,1.5938819) q[2];
u(-1.4477542e-10,0.019814817,0.91696514) q[3];
u(pi/2,0.01626071,2.1809905) q[4];
