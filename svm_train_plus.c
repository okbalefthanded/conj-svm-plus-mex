// This implementation is a MEX extension of libsvm and implements alternating SMO and conjugate
// alternating SMO for solving SVM+, described in [1] and [2].
// Okba BEKHELIFI <okba.bekhelifi@univ-usto.dz>
// Original C/C++ code by : Dmitry Pechyony, pechyony@gmail.com
// [1] D. Pechyony, R. Izmailov, A. Vashist and V. Vapnik. SMO-style Algorithms for Learning using Privileged Information . DMIN 2010.
// [2] D. Pechyony and V. Vapnik. Fast Optimization Algorithms for Solving SVM+. Chapter in Statistical Learning and Data Science, 2011.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
//
#include "svm.h"
//
#include "mex.h"
#include "svm_model_matlab.h"

#ifdef MX_API_VER
#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#endif

#define CMD_LEN 2048
#define Malloc(type, n) (type *)malloc((n) * sizeof(type))

void print_null(const char *s)
{
}
void print_string_matlab(const char *s) { mexPrintf(s); }

void exit_with_help()
{
	mexPrintf(
		"Usage: model = svmtrain(training_label_vector, training_instance_matrix, privileged_information_matrix,'libsvm_options');\n"
		"options:\n"
		"-s svm_type : set type of SVM (default 0)\n"
		"	0 -- C-SVC\n"
		"	1 -- nu-SVC\n"
		"	2 -- one-class SVM\n"
		"	3 -- epsilon-SVR\n"
		"	4 -- nu-SVR\n"
		"	5 -- SVM+\n"
		"-a n : optimization method \n"
		"    -1  -- Max Unconstrained Gain SMO (default)\n"
		"     0  -- Max Constrained Gain SMO (Glassmachers&Igel, JMLR2006)\n"
		"    k>0 -- Conjugate SMO of order k\n"
		"-t kernel_type : set type of kernel function (default 2)\n"
		"	0 -- linear: u'*v\n"
		"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
		"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
		"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
		"	4 -- precomputed kernel (kernel values in training_set_file)\n"
		"-T kernel_type_star : set type of kernel function for the correcting space (default 2), for SVM+\n"
		"	0 -- linear: u'*v\n"
		"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
		"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
		"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
		"	4 -- precomputed kernel (kernel values in training_set_file)\n"
		"-f star_file : name of the file containing star examples. Necessary parameter for SVM+ \n"
		"-d degree : set degree in kernel function (default 3)\n"
		"-D degree_star : set degree_star in kernel function in the correcting space (default 3)\n"
		"-g gamma : set gamma in kernel function (default 1/number of features)\n"
		"-G gamma_star : set gamma_star in kernel function in the correcting space (default 1/number of features in the  correcting space)\n"
		"-r coef0 : set coef0 in kernel function (default 0)\n"
		"-R coef0_star : set coef0_star in kernel function (default 0)\n"
		"-c cost : set the parameter C of C-SVC, epsilon-SVR, nu-SVR and SVM+ (default 1)\n"
		"-C tau : set the parameter tau in SVM+ (default 1)\n"
		"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
		"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
		"-m cachesize : set cache memory size in MB (default 100)\n"
		"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
		"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
		"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
		"-wi weight : set the parameter C of class i to weight*C, for C-SVC and SVM+ (default 1)\n"
		"-v n: n-fold cross validation mode\n"
		"-q : quiet mode (no outputs)\n");
}

struct svm_parameter param;   // set by parse_command_line
struct svm_problem prob;	  // set by read_problem
struct svm_problem prob_star; // set by read_problem
struct svm_model *model;
struct svm_node *x_space;
struct svm_node *x_space_star;
int cross_validation;
int nr_fold;

// read in a problem (svmlight format)
int read_problem_dense(const mxArray *label, const mxArray *instance_mat, struct svm_problem *prob, struct svm_node **x_space, int *prob_max_index)
{

	// mexPrintf("inside read_problem_dense \n");

	//using size_t due to the output type of matlab functions
	size_t i, j, k, l;
	size_t elements, max_index, sc, label_vector_row_num;
	double *samples, *labels;

	labels = mxGetPr(label);
	samples = mxGetPr(instance_mat);
	sc = mxGetN(instance_mat);
	elements = 0;

	// number of instances
	l = mxGetM(instance_mat);
	label_vector_row_num = mxGetM(label);
	prob->l = (int)l;

	if (label_vector_row_num != l)
	{
		mexPrintf("Length of label vector does not match # of instances \n");
		return -1;
	}

	for (i = 0; i < l; i++)
	{
		for (k = 0; k < sc; k++)
		{
			if (samples[k * l + i] != 0)
			{
				//mexPrintf("samples %f\n", samples[k * l + i]);
				elements++;
			}
			// count the '-1' elements
			elements++;
		}
	}

	prob->y = Malloc(double, l);
	prob->x = Malloc(struct svm_node *, l);
	*x_space = Malloc(struct svm_node, elements);

	max_index = sc;
	j = 0;
	for (i = 0; i < l; i++)
	{
		prob->x[i] = &((*x_space)[j]);
		prob->y[i] = labels[i];

		for (k = 0; k < sc; k++)
		{
			if (samples[k * l + i] != 0)
			{
				(*x_space)[j].index = (int)k + 1;
				(*x_space)[j].value = samples[k * l + i];
				j++;
			}
		}
		(*x_space)[j++].index = -1;
	}

	*prob_max_index = max_index;
	return 0;
}

int read_problem_sparse(const mxArray *label_vec, const mxArray *instance_mat, struct svm_problem *prob, struct svm_node **x_space, int *prob_max_index)
{

	// mexPrintf("Reading a sparse problem\n");
	mwIndex *ir, *jc, low, high, k;
	// using size_t due to the output type of matlab functions
	size_t i, j, l, elements, max_index, label_vector_row_num;
	mwSize num_samples;
	double *samples, *labels;
	mxArray *instance_mat_col; // transposed instance sparse matrix

	// transpose instance matrix
	{
		mxArray *prhs[1], *plhs[1];
		prhs[0] = mxDuplicateArray(instance_mat);
		if (mexCallMATLAB(1, plhs, 1, prhs, "transpose"))
		{
			mexPrintf("Error: cannot transpose training instance matrix\n");
			return -1;
		}
		instance_mat_col = plhs[0];
		mxDestroyArray(prhs[0]);
	}

	// each column is one instance
	labels = mxGetPr(label_vec);
	samples = mxGetPr(instance_mat_col);
	ir = mxGetIr(instance_mat_col);
	jc = mxGetJc(instance_mat_col);

	num_samples = mxGetNzmax(instance_mat_col);

	// number of instances
	l = mxGetN(instance_mat_col);
	label_vector_row_num = mxGetM(label_vec);
	prob->l = (int)l;

	if (label_vector_row_num != l)
	{
		mexPrintf("Length of label vector does not match # of instances.\n");
		return -1;
	}

	elements = num_samples + l;
	max_index = mxGetM(instance_mat_col);

	prob->y = Malloc(double, l);
	prob->x = Malloc(struct svm_node *, l);
	*x_space = Malloc(struct svm_node, elements);

	j = 0;
	for (i = 0; i < l; i++)
	{

		prob->x[i] = &((*x_space)[j]);
		prob->y[i] = labels[i];
		low = jc[i], high = jc[i + 1];
		for (k = low; k < high; k++)
		{
			(*x_space)[j].index = (int)ir[k] + 1;
			(*x_space)[j].value = samples[k];
			j++;
		}
		(*x_space)[j++].index = -1;
	}

	*prob_max_index = max_index;

	return 0;
}
static void fake_answer(int nlhs, mxArray *plhs[])
{
	int i;
	for (i = 0; i < nlhs; i++)
		plhs[i] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

// TODO
double do_cross_validation()
{

	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);
	double retval = 0.0;

	mexPrintf("Doing Cross validation \n");

	svm_cross_validation(&prob, &param, nr_fold, target);

	if (param.svm_type == EPSILON_SVR ||
		param.svm_type == NU_SVR)
	{
		for (i = 0; i < prob.l; i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v - y) * (v - y);
			sumv += v;
			sumy += y;
			sumvv += v * v;
			sumyy += y * y;
			sumvy += v * y;
		}
		mexPrintf("Cross Validation Mean squared error = %f\n", total_error / prob.l);
		mexPrintf("Cross Validation Squared correlation coefficient = %f\n",
				  ((prob.l * sumvy - sumv * sumy) * (prob.l * sumvy - sumv * sumy)) /
					  ((prob.l * sumvv - sumv * sumv) * (prob.l * sumyy - sumy * sumy)));
		retval = total_error / prob.l;
	}
	else
	{
		for (i = 0; i < prob.l; i++)
			if (target[i] == prob.y[i])
				++total_correct;
		mexPrintf("Cross Validation Accuracy = %f%%\n", 100.0 * total_correct / prob.l);
		retval = 100.0 * total_correct / prob.l;
	}
	free(target);

	return retval;
}

// nrhs should be 4
int parse_command_line(int nrhs, const mxArray *prhs[], char *model_file_name)
{

	// mexPrintf("In : Parse Command Line\n");
	int i, argc = 1;
	char cmd[CMD_LEN];
	char *argv[CMD_LEN / 2];
	void (*print_func)(const char *) = print_string_matlab; //default printing to matlab display

	// mexPrintf("cmd %s\n", cmd);
	// mexPrintf("argv %s\n", argv);

	// mexPrintf("about to set default param vals\n");
	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.kernel_type_star = RBF;
	param.degree = 3;
	param.degree_star = 3;
	param.gamma = 0;	  // 1/k (k: num_features)
	param.gamma_star = 0; // 1/k (k: num_features)
	param.coef0 = 0;
	param.coef0_star = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.tau = 0;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.optimizer = -1;
	cross_validation = 0;
	//input_file_name_star[0] = '\0';

	if (nrhs <= 1)
		return 1;

	if (nrhs > 3)
	{

		mxGetString(prhs[3], cmd, mxGetN(prhs[3]) + 1);
		if ((argv[argc] = strtok(cmd, " ")) != NULL)
			while ((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
	}

	// mexPrintf("Parsing options\n");
	// parse options
	for (i = 1; i < argc; i++)
	{
		if (argv[i][0] != '-')
			break;
		i++;
		if (i >= argc && argv[i - 1][1] != 'q') // since option -q has no parameter
			return;
		switch (argv[i - 1][1])
		{
		case 'a':
			param.optimizer = atoi(argv[i]);
			break;
		case 's':
			param.svm_type = atoi(argv[i]);
			break;
		case 't':
			param.kernel_type = atoi(argv[i]);
			break;
		case 'T':
			param.kernel_type_star = atoi(argv[i]);
			break;
		case 'd':
			param.degree = atoi(argv[i]);
			break;
		case 'D':
			param.degree_star = atoi(argv[i]);
			break;
		case 'g':
			param.gamma = atof(argv[i]);
			break;
		case 'G':
			param.gamma_star = atof(argv[i]);
			break;
		case 'r':
			param.coef0 = atof(argv[i]);
			break;
		case 'R':
			param.coef0_star = atof(argv[i]);
			break;
		case 'n':
			param.nu = atof(argv[i]);
			break;
		case 'm':
			param.cache_size = atof(argv[i]);
			break;
		case 'c':
			param.C = atof(argv[i]);
			break;
		case 'C':
			param.tau = atof(argv[i]);
			break;
		case 'e':
			param.eps = atof(argv[i]);
			break;
		case 'p':
			param.p = atof(argv[i]);
			break;
		case 'h':
			param.shrinking = atoi(argv[i]);
			break;
		case 'b':
			param.probability = atoi(argv[i]);
			break;
		case 'q':
			print_func = &print_null;
			i--;
			break;
		case 'v':
			cross_validation = 1;
			nr_fold = atoi(argv[i]);
			if (nr_fold < 2)
			{
				mexPrintf("n-fold cross validation: n must >= 2\n");
				return 1;
			}
			break;
		case 'w':
			++param.nr_weight;
			param.weight_label = (int *)realloc(param.weight_label, sizeof(int) * param.nr_weight);
			param.weight = (double *)realloc(param.weight, sizeof(double) * param.nr_weight);
			param.weight_label[param.nr_weight - 1] = atoi(&argv[i - 1][2]);
			param.weight[param.nr_weight - 1] = atof(argv[i]);
			break;
		/*
	case 'f':
	  // fix it, dont read file but pass a matrix as input
      strcpy(input_file_name_star, argv[i]);
	  break;
		*/
		default:
			mexPrintf("Unknown option: -%c\n", argv[i - 1][1]);
			return 1;
		}
	}
	svm_set_print_string_function(print_func);

	return 0;
}

// Interface function of MATLAB
// now assume prhs[0]: label prhs[1]: features prhs[2] priviliged information prhs[3] options
void mexFunction(int nlhs, mxArray *plhs[],
				 int nrhs, const mxArray *prhs[])
{
	//
	const char *name = mexFunctionName();
	// mexPrintf("%s called with (%d) inputs, (%d) outputs \n", name, nrhs, nlhs);

	const char *error_msg;
	int max_index;

	// fix random seed to have same results for each run
	// (for cross validation and probability estimation
	srand(1);

	// output > 1
	if (nlhs > 1)
	{
		exit_with_help();
		fake_answer(nlhs, plhs);
		return;
	}

	// mexPrintf("transforming inputs...\n");
	// Transform the input Matrix to libsvm format
	// inputs
	if (nrhs > 1 && nrhs < 5)
	{
		int err;

		if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]))
		{
			mexPrintf("testing entries are double\n");
			mexPrintf("Error: label vector and instance matrix must be double\n");
			fake_answer(nlhs, plhs);
			return;
		}

		if (mxIsSparse(prhs[0]))
		{
			mexPrintf("testing labels are sparse\n");
			mexPrintf("Error: label vector should not be in a sparse format\n");
			fake_answer(nlhs, plhs);
			return;
		}

		if (parse_command_line(nrhs, prhs, NULL))
		{
			exit_with_help();
			svm_destroy_param(&param);
			fake_answer(nlhs, plhs);
			return;
		}

		// training_instance_matrix
		if (mxIsSparse(prhs[1]))
		{
			// mexPrintf("Testing if kernel is PRECOMPUTED \n");
			if (param.kernel_type == PRECOMPUTED)
			{
				// precomputed kernel requires dense matrix, so we make one
				mxArray *rhs[1], *lhs[1];
				rhs[0] = mxDuplicateArray(prhs[1]);
				
				if (mexCallMATLAB(1, lhs, 1, rhs, "full"))
				{
					mexPrintf("Error: cannot generate a full training instance matrix");
					svm_destroy_param(&param);
					fake_answer(nlhs, plhs);
					return;
				}
				// mexPrintf("Reading Dense Problem for PRCOMPUTED kernel \n");
				// err = read_problem_dense(prhs[0], lhs[0]);
				err = read_problem_dense(prhs[0], lhs[0], &prob, &x_space, &max_index);
				//err = read_problem_dense(prhs[0], lhs[0], &prob, &x_space);
				check_kernel_input(prob, max_index);

				mxDestroyArray(lhs[0]);
				mxDestroyArray(rhs[0]);
			}
			else
			{
				// prhs[0]: label, prhs[1]: instance_matrix
				// mexPrintf("Reading a sparse problem \n");
				err = read_problem_sparse(prhs[0], prhs[1], &prob, &x_space, &max_index);
				// mexPrintf("Train Data Reading problem sparse return:  %d\n", err);
			}
		}
		else
		{
			// mexPrintf("Reading Dense Problem \n");
			// err = read_problem_dense(prhs[0], prhs[1]);
			err = read_problem_dense(prhs[0], prhs[1], &prob, &x_space, &max_index);
			// mexPrintf("Train Data Reading problem dense return:  %d\n", err);
		}
		if (param.gamma == 0 && max_index > 0)
			param.gamma = 1.0 / max_index;

		// svmtrain's original code
		error_msg = svm_check_parameter(&prob, &param);

		if (err || error_msg)
		{
			if (error_msg != NULL)
				mexPrintf("Error: %s\n", error_msg);
			svm_destroy_param(&param);
			free(prob.y);
			free(prob.x);
			free(x_space);
			fake_answer(nlhs, plhs);
			return;
		}
		//bool r = check_compatibility(prob, prob_star);
		// mexPrintf("Testing Privileged Information \n");
		// privileged_information_matrix
		// prhs[0]: label, prhs[2]: privileged_information
		if (mxIsSparse(prhs[2]))		{

			// mexPrintf("Testing if kernel is PRECOMPUTED \n");
			if (param.kernel_type_star == PRECOMPUTED)
			{
				// precomputed kernel requires dense matrix, so we make one
				mxArray *rhs[1], *lhs[1];
				rhs[0] = mxDuplicateArray(prhs[2]);
				if (mexCallMATLAB(1, lhs, 1, rhs, "full"))
				{
					mexPrintf("Error: cannot generate a full training instance matrix");
					svm_destroy_param(&param);
					fake_answer(nlhs, plhs);
					return;
				}
				// mexPrintf("Reading Dense Problem for PRCOMPUTED kernel \n");
				// err = read_problem_dense(prhs[0], lhs[0]);
				err = read_problem_dense(prhs[0], prhs[2], &prob_star, &x_space_star, &max_index);
				//err = read_problem_dense(prhs[0], lhs[0], &prob, &x_space);
				check_kernel_input(prob_star, max_index);
				mxDestroyArray(lhs[0]);
				mxDestroyArray(rhs[0]);
			}
			else
			{
				// prhs[0]: label, prhs[1]: instance_matrix
				// mexPrintf("Reading a sparse problem \n");
				err = read_problem_sparse(prhs[0], prhs[2], &prob_star, &x_space_star, &max_index);
				// mexPrintf("Privileged info Reading problem sparse return:  %d\n", err);
			}
		}
		else
		{
			// mexPrintf("Reading Dense Problem \n");
			// err = read_problem_dense(prhs[0], prhs[1]);
			err = read_problem_dense(prhs[0], prhs[2], &prob_star, &x_space_star, &max_index);
			// mexPrintf("Privileged info Reading problem dense return:  %d\n", err);
		}
		if (param.gamma_star == 0 && max_index > 0)
			param.gamma_star = 1.0 / max_index;
		
		if (check_compatibility(prob, prob_star) == 1)
			{
				
				fake_answer(nlhs, plhs);
					return;
			}
		
		prob.x_star = prob_star.x; /* merge two problems into a single one */

		error_msg = svm_check_parameter(&prob_star, &param);

		if (err || error_msg)
		{
			if (error_msg != NULL)
				mexPrintf("Error: %s\n", error_msg);
			svm_destroy_param(&param);
			free(prob_star.y);
			free(prob_star.x);
			free(x_space_star);
			fake_answer(nlhs, plhs);
			return;
		}

		// mexPrintf("About to cross validate \n");
		if (cross_validation)
		{
			double *ptr;
			plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
			ptr = mxGetPr(plhs[0]);
			// mexPrintf("Doing cross validation \n");
			ptr[0] = do_cross_validation();
		}

		else
		{
			
			int nr_feat = (int)mxGetN(prhs[1]);
			const char *error_msg;
			model = svm_train(&prob, &param);
			
			error_msg = model_to_matlab_structure(plhs, nr_feat, model);

			if (error_msg)
				mexPrintf("Error: can't convert libsvm model to matrix structure: %s\n", error_msg);
			svm_destroy_model(&model);
		}
		svm_destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
		free(prob_star.y);
		free(prob_star.x);
		free(x_space_star);
	}
	else
	{
		mexPrintf("No Input \n");
		exit_with_help();
		fake_answer(nlhs, plhs);
		return;
	}

}
