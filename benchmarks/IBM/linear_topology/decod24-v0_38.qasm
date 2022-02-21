OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
u(1.570795,5.8020142,-pi) q[1];
u(pi,-4.8754758,2.9332918) q[2];
cz q[1],q[2];
u(pi/2,4.1187286,pi/2) q[0];
u(1.5707977,-pi/2,-5.7390422) q[1];
cz q[0],q[1];
u(pi/2,-2.6937279,4.5493021) q[2];
u(pi/2,-1.7705093,pi/2) q[3];
cz q[2],q[3];
u(pi/2,-5.1319946,pi/2) q[1];
u(pi/2,-3*pi/4,2.6937279) q[2];
cz q[1],q[2];
u(pi/2,-4.8210086,-4.1187286) q[0];
u(pi,-4.8924762,-0.62459789) q[1];
cz q[0],q[1];
u(pi/2,-4.3653535,3*pi/2) q[2];
u(pi,-1.1067614,-3.3413057) q[3];
cz q[2],q[3];
u(0.95531665,4.9726388,-1.6495144) q[1];
u(0.015284076,3.1435932,2.7925564) q[2];
cx q[1],q[2];
u(-1.0512475,0.068198518,-0.13676985) q[2];
cx q[1],q[2];
u(-1.4385811,1.6664412,0.015853959) q[2];
u(1.5191595,-3.2306427,-0.12976847) q[3];
cz q[2],q[3];
u(pi/2,2.514153,-0.21534012) q[2];
u(pi/2,-1.7897141,2.4465801) q[3];
cz q[2],q[3];
u(-5.3278687,1.8788831,-2.8782438) q[1];
u(1.6224331,0.090088467,2.9849696) q[2];
cz q[1],q[2];
u(pi,2.8179911,0.23961329) q[0];
u(1.6721319,-0.24788485,-5.8058739) q[1];
cz q[0],q[1];
u(pi/2,-4.5513892,-1.1162011) q[0];
u(pi/2,-1.5708179,1.8186812) q[1];
cz q[0],q[1];
u(pi/2,-1.7078615,4.6740063) q[2];
u(pi,-0.44167761,-3.2701098) q[3];
cz q[2],q[3];
u(pi/2,-0.1013356,1.4097966) q[0];
u(0.062972929,1.5708396,-2.160902e-05) q[1];
u(1.5452101e-08,1.716343e-05,-1.4337484) q[2];
u(pi/2,-pi/2,1.6650389) q[3];
