OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
u(pi/2,-1.870093,-4.7947486) q[1];
u(pi/2,-3.6008799,-2.9609912) q[2];
cz q[1],q[2];
u(pi/2,5.0196217,pi) q[0];
u(pi/2,-0.1825279,-1.2714997) q[1];
cz q[0],q[1];
u(pi/2,-0.96086752,1.7533242) q[1];
u(pi/2,0.57427166,0.4592873) q[2];
cz q[1],q[2];
u(pi/4,-2.3098953,2.567321) q[2];
u(pi,1.3104456,-4.7716897) q[3];
cz q[2],q[3];
u(-pi/8,-1.0586196,0.96086941) q[1];
u(pi/2,-0.17664815,-5.5440865) q[2];
cz q[1],q[2];
u(-pi,5.7194768,0.46000636) q[0];
u(5*pi/8,-1.8160088,1.0586179) q[1];
cz q[0],q[1];
u(7*pi/8,-0.18628474,1.8160104) q[1];
u(pi/2,-4.2347489,0.045479636) q[2];
cz q[1],q[2];
u(pi/2,2.1570391,1.0931563) q[2];
u(2.0742808,-3.1805929,0.32072364) q[3];
cz q[2],q[3];
u(pi/2,-0.34823615,3.3278792) q[1];
u(1.6280466e-08,-1.5707944,4.1624424) q[2];
cz q[1],q[2];
u(-pi,-4.5096621,-4.2167826) q[0];
u(1.8323269,-5.3851693e-06,5.0606252) q[1];
cz q[0],q[1];
u(pi/2,-3.8290142,3.1052952) q[2];
u(pi/2,-2.2580974,0.67025863) q[3];
cz q[2],q[3];
u(pi/2,-1.4511092,1.5708016) q[1];
u(2.0742808,-2.24264,-1.8229128) q[2];
cz q[1],q[2];
u(1.6428798,0.97241456,1.5870034) q[2];
u(1.4226476,-0.0031816437,0.70885382) q[3];
cx q[2],q[3];
u(-2.3537798,0.0084653625,0.033423749) q[3];
cx q[2],q[3];
u(5*pi/2,-1.2004824,-1.2977844) q[1];
u(1.6428798,1.5685567,4.5227656) q[2];
cz q[1],q[2];
u(-1.1383437e-07,0.010956232,3.5623491) q[0];
u(3.0708279,0.028433735,-0.34180897) q[1];
cx q[0],q[1];
u(-0.7854029,-0.0036784993,0.0031870753) q[1];
cx q[0],q[1];
u(2.9259925e-08,-5.4778772,-1.0178561) q[2];
u(1.9955747,-4.4320392,3.1591192) q[3];
cz q[2],q[3];
u(pi/2,-9.4435201e-08,5.2061036) q[0];
u(1.5699676,0.011623499,-pi/2) q[1];
u(4.1818265e-08,-5.3572903e-06,-0.57060528) q[2];
u(pi/2,1.5808844,-0.23784363) q[3];
u(0,8.7458285e-10,-2.1235156e-09) q[4];