OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
u(pi/2,4.2138558,3*pi/2) q[2];
u(pi/2,-2.8911994,6.6684641e-09) q[1];
cz q[2],q[1];
u(pi/2,-1.6364977,0.49853326) q[2];
u(1.7469874,-1.4636013,0.43651975) q[0];
cx q[2],q[0];
u(1.8171836,-0.060601494,-0.026342892) q[0];
cx q[2],q[0];
u(-pi,1.281587,-0.17867015) q[3];
u(1.5490725,pi/2,-3.881176) q[2];
cx q[3],q[2];
u(-2.3534853,-0.07661557,-0.086712909) q[2];
cx q[3],q[2];
u(-3.1917851,4.6776222,1.9783322) q[2];
u(2.30637,-4.1965659,1.9933375) q[0];
cz q[2],q[0];
u(pi/2,0.12846031,-2.1803345e-09) q[4];
u(0.35376569,-3.304381,0.44184372) q[2];
cz q[4],q[2];
u(pi/2,0.66455206,-3.2700529) q[4];
u(3*pi/4,0.35784235,-0.25039327) q[1];
cz q[4],q[1];
u(pi/2,3.1337418,-3.8061447) q[4];
u(pi,-4.7573478,-1.5916839) q[3];
cz q[4],q[3];
u(pi/2,5.4409907,-3.1337418) q[4];
u(1.8504956,0.98337698,-0.066422285) q[1];
cx q[4],q[1];
u(2.0889893,-0.10269683,-0.091366697) q[1];
cx q[4],q[1];
u(pi,-4.1630811,-4.2928638) q[4];
u(pi/2,-pi/2,1.7335853) q[2];
cz q[4],q[2];
u(pi/2,2.4267823,-2.5427737) q[4];
u(pi/2,0.080941229,-3.0603353) q[0];
cz q[4],q[0];
u(-1.2788795,-2.028166,-2.4760747) q[1];
u(pi/2,-1.5705718,3.0606514) q[0];
cz q[1],q[0];
u(pi/2,-1.098405,3*pi/2) q[2];
u(3*pi/4,-0.00015866499,3.1413682) q[0];
cz q[2],q[0];
u(0,-0.044958839,1.9880464) q[3];
u(pi/4,-pi,0.18206576) q[1];
cz q[3],q[1];
u(pi/2,4.5891456,3.8564031) q[4];
u(3*pi/4,-3.1415851,0.0001587319) q[0];
cz q[4],q[0];
u(pi/2,2.8215203,5.6210306) q[4];
u(pi/2,0.45173263,-4.3993821) q[2];
cz q[4],q[2];
u(pi/2,-pi/4,1.5707888) q[0];
u(pi/2,-pi/4,3*pi/2) q[1];
u(pi/2,-pi/2,2.68986) q[2];
u(0,-0.044958839,-4.340791) q[3];
u(6.5299528e-10,5.299475e-06,-1.2507293) q[4];
