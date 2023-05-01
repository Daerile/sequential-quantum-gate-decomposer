# include <math.h>
# include <interval.h>
# include <vector>
# include <stdio.h>
# include <iostream>
# include <QJsonDocument>
# include <QJsonObject>
# include <QFile>
# include <QTextStream>

#include <gsl/gsl_vector.h>
#include "N_Qubit_Decomposition_adaptive.h"
#include "Decomposition_Base.h"
#include "Gates_block.h"
#include "matrix_real.h"
#include <math.h>

//../bin/OptimusApp --filename=libsquander.so --opt_method=Pso --pso_particles=100 --pso_generations=10

using namespace std;

extern "C"
{
N_Qubit_Decomposition_adaptive* decomp = NULL;
double cur_min;
string qasm;
int cntnue = 0;

void    init(QJsonObject data)
{
    //printf("%s\n", QJsonDocument(data).toJson(QJsonDocument::Compact).toStdString().c_str());
    if (decomp == NULL) {
        string folder = data.take("folder").toString().toStdString();
        qasm = data.take("qasm").toString().toStdString();        
        int qbit_num, accelerator_num = 0, level_limit=data.take("levels").toString().toInt(), level_limit_min=3;
        if (system((std::string("python ../../sequential-quantum-gate-decomposer/saveunitary.py ") + folder + qasm + ".qasm ./" + qasm + ".binary").c_str()) != 0) {
            exit(-1);
        }        
        std::string filename = qasm + ".binary";
        Matrix Umtx = Decomposition_Base().import_unitary_from_binary(filename);
        qbit_num = log2(Umtx.rows);
        //Matrix Umtx = create_identity( 1 << qbit_num);
        decomp = new N_Qubit_Decomposition_adaptive( Umtx, qbit_num, level_limit, level_limit_min, accelerator_num );
        if (data.contains("continue")) {
            cntnue = 1;
            Matrix_real parameters_mtx;
            decomp->combine(import_gate_list_from_binary(parameters_mtx, qasm + ".gates"));
            Matrix Umtx_new = decomp->get_transformed_matrix( parameters_mtx, decomp->get_gates().begin(), decomp->get_gates().size(), Umtx );
            printf("%f %d %d\n", decomp->optimization_problem(parameters_mtx), parameters_mtx.size(), decomp->get_gates().size());
            delete decomp;
            decomp = new N_Qubit_Decomposition_adaptive( Umtx_new, qbit_num, level_limit, level_limit_min, accelerator_num );
            decomp->add_adaptive_layers();
        } else {
            //Matrix_real optimized_parameters_mtx_loc;
            //Gates_block* gate_structure_loc = decomp->determine_initial_gate_structure(optimized_parameters_mtx_loc);
            //decomp->combine( gate_structure_loc );
            for (int idx=0; idx<level_limit; idx++ ) {
                  decomp->add_adaptive_layers();
            }
        }
        decomp->add_finalyzing_layer();
        cur_min = 1.0;
    }
}

int	getdimension()
{
	return decomp->get_parameter_num();
}
void    getmargins(vector<Interval> &x)
{
        for(size_t i=0;i<x.size();i++)
                x[i]=Interval(0,2*M_PI);
}


//f(x)
double	funmin(vector<double> &x)
{
   return decomp->optimization_problem(x.data());
   //Matrix_real parameters_mtx(x.data(), 1, x.size());
   //return decomp->optimization_problem(parameters_mtx);
}

//f'(x)
void    granal(vector<double> &x,vector<double> &g)
{
    gsl_vector* grad_gsl = gsl_vector_alloc(decomp->get_parameter_num());
    gsl_vector gv = { x.size(), sizeof(double), x.data(), NULL, 0 };
    decomp->optimization_problem_grad(&gv, (void*)decomp, grad_gsl);
    for (int i = 0; i < decomp->get_parameter_num(); i++) {
        g[i] = gsl_vector_get(grad_gsl, i);
    }
    gsl_vector_free(grad_gsl);
}

QJsonObject    done(vector<double> &x)
{
   double min = decomp->optimization_problem(x.data());
   if (min < cur_min) { //the done function should be single-threaded and hence thread-safe
       cur_min = min;
       //printf("New Min: %f\n", cur_min);
       if (cntnue) {
            //decomp->add_adaptive_gate_structure(qasm + ".gates");
            Matrix_real parameters_mtx;
            Gates_block* gb = import_gate_list_from_binary(parameters_mtx, qasm + ".gates");
            printf("%d %d %d\n", parameters_mtx.size(), gb->get_gates().size(), x.size());
            Matrix_real combine_parameters_mtx(1, parameters_mtx.size() + x.size());
            memcpy(combine_parameters_mtx.get_data(), parameters_mtx.get_data(), parameters_mtx.size() * sizeof(double));
            memcpy(combine_parameters_mtx.get_data() + parameters_mtx.size(), x.data(), x.size() * sizeof(double)); 
            gb->combine(decomp);
            printf("%d %d\n", combine_parameters_mtx.size(), gb->get_gates().size());
            export_gate_list_to_binary(combine_parameters_mtx, gb, qasm + ".next.gates");
       } else {
           Matrix_real parameters_mtx(x.data(), 1, x.size());
           export_gate_list_to_binary(parameters_mtx, decomp, qasm + ".next.gates");
       }
   }
    //printf("done\n");
    /*
    delete decomp;
    decomp = NULL;*/
    return QJsonObject();
}
}
