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
#include <mutex>

//../bin/OptimusApp --filename=libsquander.so --opt_method=Pso --pso_particles=100 --pso_generations=10

using namespace std;

extern "C"
{
N_Qubit_Decomposition_adaptive* decomp = NULL;
double cur_min;
string qasm;
mutex min_mutex;
int cntnue = 0;

void savemin(double min, vector<double> &x) {
   lock_guard<mutex> guard(min_mutex);
   if (min < cur_min) { //the done function should be single-threaded and hence thread-safe
       cur_min = min;
       printf("New Min: %f\n", cur_min);
       if (cntnue) {
            //decomp->add_adaptive_gate_structure(qasm + ".gates");
            Matrix_real parameters_mtx;
            Gates_block* gbi = import_gate_list_from_binary(parameters_mtx, qasm + ".gates", 0);
            Gates_block* gb = decomp->clone(); gb->combine(gbi); delete gbi;
            //printf("%d %d %d %d\n", parameters_mtx.size(), gb->get_gates().size(), x.size(), gb->get_parameter_num());
            Matrix_real combine_parameters_mtx(1, parameters_mtx.size() + x.size());
            memcpy(combine_parameters_mtx.get_data(), x.data(), x.size() * sizeof(double));
            memcpy(combine_parameters_mtx.get_data() + x.size(), parameters_mtx.get_data(), parameters_mtx.size() * sizeof(double)); 
            //gb->combine(decomp);
            //gb->list_gates(combine_parameters_mtx, 0);
            //combine_parameters_mtx.print_matrix();
            //printf("%d %d %d\n", combine_parameters_mtx.size(), gb->get_gates().size(), gb->get_parameter_num());
            export_gate_list_to_binary(combine_parameters_mtx, gb, qasm + ".next.gates");
            delete gb;
       } else {
           Matrix_real parameters_mtx(x.data(), 1, x.size());
           export_gate_list_to_binary(parameters_mtx, decomp, qasm + ".next.gates");
       }
       if (cur_min < 1e-8) exit(0);
   }
}

void    init(QJsonObject data)
{
    //printf("%s\n", QJsonDocument(data).toJson(QJsonDocument::Compact).toStdString().c_str());
    if (decomp == NULL) {
        string folder = data.take("folder").toString().toStdString();
        qasm = data.take("qasm").toString().toStdString();        
        int qbit_num, accelerator_num = 0, level_limit=data.take("levels").toString().toInt(), level_limit_min=1;
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
            Gates_block* gb = import_gate_list_from_binary(parameters_mtx, qasm + ".gates", 0);
            decomp->combine(gb);
            delete gb;
            Matrix Umtx_new = decomp->get_transformed_matrix( parameters_mtx, decomp->get_gates().begin(), decomp->get_gates().size(), Umtx );
            //printf("%f %d %d\n", decomp->optimization_problem(parameters_mtx), parameters_mtx.size(), decomp->get_gates().size());
            delete decomp;
            decomp = new N_Qubit_Decomposition_adaptive( Umtx_new, qbit_num, level_limit, level_limit_min, accelerator_num );
        }
        //decomp->set_cost_function_variant(SUM_OF_SQUARES);
        //Matrix_real optimized_parameters_mtx_loc;
        //Gates_block* gate_structure_loc = decomp->determine_initial_gate_structure(optimized_parameters_mtx_loc);
        //decomp->combine( gate_structure_loc );
        for (int idx=0; idx<level_limit; idx++ ) {
              decomp->add_adaptive_layers();
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
   double value = decomp->optimization_problem(x.data());
   if (value < cur_min) savemin(value, x);
   return value;
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
   savemin(decomp->optimization_problem(x.data()), x);
    //printf("done\n");
    /*
    delete decomp;
    decomp = NULL;*/
    return QJsonObject();
}
}
