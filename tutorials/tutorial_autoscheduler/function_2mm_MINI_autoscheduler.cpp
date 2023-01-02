#include <tiramisu/tiramisu.h>
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function_2mm_MINI_wrapper.h"


using namespace tiramisu;

const std::string py_cmd_path = "/usr/bin/python";
const std::string py_interface_path = "/home/afif/multi/tiramisu/tutorials/tutorial_autoscheduler/model/main.py";
int main(int argc, char **argv)

{
    tiramisu::init("function_2mm_MINI");

    // -------------------------------------------------------
    // Layer I
    // ------------------------------------------------------- 
     var i("i", 0, 16), j("j", 0, 32), m("m", 0, 32), k("k", 0, 32), l("l", 0, 16);

    //inputs
    input A("A", {i, k}, p_float64);
    input B("B", {k, l}, p_float64);
    input C("C", {l, j}, p_float64);
    input D("D", {i, j}, p_float64);
    
    //Computations
    computation D_init("D_init", {i,m}, D(i,m)*1.2);
    computation D_out("D_out", {i,j,k,l}, p_float64);
    D_out.set_expression(D(i,j) + A(i,k)*B(k,l)*C(l,j)*1.5);

    // -------------------------------------------------------
    // Layer II
    // -------------------------------------------------------
     D_init.then(D_out, i);
    
    // -------------------------------------------------------
    // Layer III
    // -------------------------------------------------------
    //Input Buffers
    buffer b_D("b_D", {16,32}, p_float64, a_output);
    buffer b_A("b_A", {16,32}, p_float64, a_input);
    buffer b_B("b_B", {32,16}, p_float64, a_input);
    buffer b_C("b_C", {16,32}, p_float64, a_input);

    //Store inputs
    A.store_in(&b_A);
    B.store_in(&b_B);
    C.store_in(&b_C);
    D.store_in(&b_D);

    //Store computations
    D_init.store_in(&b_D);
    D_out.store_in(&b_D, {i,j});
   

    // -------------------------------------------------------
    // Code Generation
    // -------------------------------------------------------
    	prepare_schedules_for_legality_checks();
	performe_full_dependency_analysis();

	const int beam_size = get_beam_size();
	const int max_depth = get_max_depth();
	declare_memory_usage();

	auto_scheduler::schedules_generator *scheds_gen = new auto_scheduler::ml_model_schedules_generator();
	auto_scheduler::evaluation_function *model_eval = new auto_scheduler::evaluate_by_learning_model(py_cmd_path, {py_interface_path});
	auto_scheduler::evaluate_by_execution *exec_eval = new auto_scheduler::evaluate_by_execution({&b_A, &b_B, &b_C, &b_D}, "function_2mm_MINI.o", "./function_2mm_MINI_wrapper");
	auto_scheduler::search_method *bs = new auto_scheduler::beam_search(beam_size, max_depth, exec_eval, scheds_gen);
	auto_scheduler::auto_scheduler as(bs, model_eval);
	as.set_exec_evaluator(exec_eval);
	as.sample_search_space("./function_2mm_MINI_explored_schedules.json", true);
	delete scheds_gen;
	delete exec_eval;
	delete bs;
	return 0;
}
