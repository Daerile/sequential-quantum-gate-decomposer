OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
u(pi/2,-1.6555331,-1*pi/8) q[3];
u(-4.5154737e-08,-0.062259312,5.4386905) q[2];
cz q[3],q[2];
u(-1.2286099e-08,-0.062194796,-2.0240421) q[2];
u(1.4705817,pi/2,-5.157056) q[1];
cx q[2],q[1];
u(-0.78048855,-0.013867863,-0.080199791) q[1];
cx q[2],q[1];
u(-5.073903,-0.33332112,-2.6363736) q[3];
u(pi/2,-pi/2,2*pi) q[0];
cz q[3],q[0];
u(-0.51730091,-2.4525527,-3*pi/2) q[4];
u(pi/2,-1.6421128,3.3180597) q[3];
cz q[4],q[3];
u(pi,-3.8961515,1.4994799) q[3];
u(pi/2,-2.5979769,3*pi/2) q[0];
cz q[3],q[0];
u(pi,-0.5272595,-5.4587866) q[4];
u(1.9086497,0.83534381,-3.1866542) q[1];
cz q[4],q[1];
u(-pi,2.8691752,-2.4271733) q[3];
u(pi/2,-1.6733586,3.7418355) q[1];
cz q[3],q[1];
u(-3.5471151,-2.8288585,-4.07143) q[3];
u(1.6210239,pi/2,-1.9233944) q[2];
cx q[3],q[2];
u(1.3234694,-0.039179657,-0.039102881) q[2];
cx q[3],q[2];
u(pi/2,-4.9185877,-2.671691) q[4];
u(1.8025173,-2.9905182,-0.83741591) q[0];
cx q[4],q[0];
u(-0.38768932,-0.051566139,-0.048874331) q[0];
cx q[4],q[0];
u(1.5206285,0.67431188,-pi/2) q[2];
u(1.0941371,-4.2497755,-2.7822355) q[0];
cz q[2],q[0];
u(pi,-5.9279751,1.2968924) q[4];
u(0.4055224,-4.4099647,1.5034237) q[3];
cz q[4],q[3];
u(1.7403432,-6.0589834,2.9991847) q[3];
u(1.6006252,pi/2,0.0089507852) q[2];
cx q[3],q[2];
u(1.2151569,-0.033359232,0.015500805) q[2];
cx q[3],q[2];
u(-3*pi,-2.7414826,3.9589021) q[4];
u(pi,-0.0061978983,-0.0061979113) q[1];
cz q[4],q[1];
u(4.9788608e-07,-3.9297855,1.8596008) q[4];
u(0.33669518,1.1949319,-0.47572874) q[0];
cx q[4],q[0];
u(-0.39727527,0.025742163,0.023950671) q[0];
cx q[4],q[0];
u(-4.8819359,3.0867125,0.99065475) q[3];
u(1.8980232,0.28412673,-4.7391018) q[0];
cz q[3],q[0];
u(-5.0778782,-0.75638526,-5.7093818) q[4];
u(0.90247077,0.017577432,-3.1751942) q[2];
cx q[4],q[2];
u(2.2781873,-0.092677481,-0.10854101) q[2];
cx q[4],q[2];
u(-4.3884027,2.7628751,3.7158228) q[3];
u(1.2255168,-3.2054577,0.01203909) q[2];
cx q[3],q[2];
u(1.3453896,0.018708069,-0.039232854) q[2];
cx q[3],q[2];
u(-4.3468992,-4.0473438,-3.2440932) q[4];
u(1.24681,-2.5306767,-1.4180219) q[3];
cz q[4],q[3];
u(5.2265574e-10,-0.010616414,4.7017704) q[0];
u(0.51730091,-pi/2,-0.1025623) q[1];
u(1.5348367,3.0792035,-pi/2) q[2];
u(pi/2,-3*pi/4,1.5852622) q[3];
u(pi/2,pi,0.5130521) q[4];
