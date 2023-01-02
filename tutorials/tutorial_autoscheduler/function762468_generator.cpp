#include <tiramisu/tiramisu.h> 
#include <tiramisu/auto_scheduler/evaluator.h>
#include <tiramisu/auto_scheduler/search_method.h>
#include "function762468_wrapper.h"

using namespace tiramisu;

int main(int argc, char **argv){                
	tiramisu::init("function762468");
	var i0("i0", 0, 512), i1("i1", 0, 256), i2("i2", 0, 256), i3("i3", 0, 256);
	input icomp00("icomp00", {i1,i2}, p_float64);
	input input01("input01", {i0,i1,i2}, p_float64);
	input icomp01("icomp01", {i1,i3}, p_float64);
	input input03("input03", {i1,i0}, p_float64);
	input input04("input04", {i3,i1,i0}, p_float64);
	computation comp00("comp00", {i0,i1,i2},  p_float64);
	comp00.set_expression((input01(i0, i1, i2) - 1)*icomp00(i1, i2));
	computation comp01("comp01", {i0,i1,i3},  p_float64);
	comp01.set_expression(expr(2.150)*(icomp01(i1, i3) + input04(i3, i1, i0))*(expr(2.832)*input03(i1, i0) + 12.398) + 0.630);
	comp00.then(comp01, i1);
	buffer buf00("buf00", {256,256}, p_float64, a_output);
	buffer buf01("buf01", {512,256,256}, p_float64, a_input);
	buffer buf02("buf02", {256,256}, p_float64, a_output);
	buffer buf03("buf03", {256,512}, p_float64, a_input);
	buffer buf04("buf04", {256,256,512}, p_float64, a_input);
	icomp00.store_in(&buf00);
	input01.store_in(&buf01);
	icomp01.store_in(&buf02);
	input03.store_in(&buf03);
	input04.store_in(&buf04);
	comp00.store_in(&buf00, {i1,i2});
	comp01.store_in(&buf02, {i1,i3});
	tiramisu::codegen({&buf00,&buf01,&buf02,&buf03,&buf04}, "function762468.o"); 
	return 0; 
}