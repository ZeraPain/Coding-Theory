#include <iostream>
#include "gurobi_c++.h"

//#define MANUAL_INPUT

void calc_lin_independ_vec(std::vector<std::vector<int>>& ret, std::vector<int>* init_vec, int q_index, const int q, int k_index, const int k)
{
	if (k_index == (k - 1))
	{
		//(*init_vec).print("c:");
		ret.push_back(*init_vec);
		return;
	}

	k_index++;
	auto construct_vec = *init_vec;
	calc_lin_independ_vec(ret, &construct_vec, q_index, q, k_index, k);

	while (construct_vec.at(k_index) < (q - 1))
	{
		construct_vec.at(k_index) += 1;
		calc_lin_independ_vec(ret, &construct_vec, q_index + 1, q, k_index, k);
	}
}

std::vector<std::vector<int>> get_lin_independ_vector(const int q, const int k)
{
	std::vector<std::vector<int>> ret;

	for (auto k_index = k - 1; k_index >= 0; --k_index)
	{
		//std::cout << "stage " << k_index << std::endl;
		std::vector<int> init_vec(k); // set size to k
		init_vec.at(k_index) = 1;
		calc_lin_independ_vec(ret, &init_vec, 1, q, k_index, k);
	}

	return ret;
}

int dot_product(const int* a, const int* b, const int size)
{
	auto ret = 0;

	for (auto i = 0; i < size; i++)
	{
		ret += a[i] * b[i];
	}

	return ret;
}

int dot_product(std::vector<int>& a, std::vector<int>& b)
{
	return dot_product(&a[0], &b[0], a.size());
}

int main()
{
	/*
	 * Schritt 1. Schreiben Sie ein Programm zur Konstruktion der
	 * Matrix Ak, q für gegebene k und q(nur Primzahlen).
	*/

#ifdef MANUAL_INPUT
	int q, k, b;

	std::cout << "Enter value for q: ";
	std::cin >> q;

	std::cout << "Enter value for k: ";
	std::cin >> k;

	std::cout << "Enter value for b: ";
	std::cin >> b;
#else
	const auto q = 11;
	const auto k = 3;
	const auto b = 10;
#endif

	auto lin_independ_vecs = get_lin_independ_vector(q, k);
	const int mat_size = (pow(q, k) - 1) / (q - 1);

	const auto A = new double*[mat_size];
	for (auto i = 0; i < mat_size; ++i)
	{
		A[i] = new double [mat_size];
	}

	for (auto i = 0; i < mat_size; ++i)
	{
		for (auto j = 0; j < mat_size; ++j)
		{
			const auto res = dot_product(lin_independ_vecs.at(i), lin_independ_vecs.at(j)) % q;
			A[i][j] = 0 == res ? 1 : 0;
		}
	}

	// print
	/*std::cout << "A:" << std::endl;
	for (auto i = 0; i < mat_size; ++i)
	{
		for (auto j = 0; j < mat_size; ++j)
		{
			std::cout << A[i][j];
		}
		std::cout << std::endl;
	}
	*/

	/*
	 * Schritt 2. Nutzen Sie GUROBI (http://www.gurobi.com), um ein
	 * Programm für das Optimierungsproblem zu schreiben.
	 */

	try
	{
		const GRBEnv env = GRBEnv();
		GRBModel model = GRBModel(env);

		model.set(GRB_StringAttr_ModelName, "Optimierungsproblem");

		std::vector<GRBVar> vars;

		for (auto i = 0; i < mat_size; i++)
		{
			std::string name = std::string("x").append(std::to_string(i));
			vars.push_back(model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER, name));
		}

		// set objective
		GRBLinExpr obj = 0.0;
		for (auto i = 0; i < mat_size; i++)
		{
			obj += vars[i];
		}

		model.setObjective(obj, GRB_MAXIMIZE);
		
		GRBLinExpr* constraints = new GRBLinExpr[mat_size];
		for (auto i = 0; i < mat_size; ++i)
		{
			for (auto j = 0; j < mat_size; ++j)
			{
				constraints[j].addTerms(&A[i][j], &vars[i], 1);
			}
		}

		for (auto i = 0; i < mat_size; ++i)
		{
			model.addConstr(constraints[i], GRB_LESS_EQUAL, b);
		}

		// Solve
		//model.update();
		//model.write("debug.lp");
		model.optimize();

		double n = 0;
		for (auto i = 0; i < mat_size; i++)
		{
			n += vars[i].get(GRB_DoubleAttr_X);
			std::cout << vars[i].get(GRB_StringAttr_VarName) << " "
				<< vars[i].get(GRB_DoubleAttr_X) << std::endl;
		}

		std::cout << "n=" << n << std::endl;

		delete[] constraints;

	}
	catch (const GRBException& e)
	{
		std::cout << "Error code = " << e.getErrorCode() << std::endl;
		std::cout << e.getMessage() << std::endl;
	}
	catch (...)
	{
		std::cout << "Exception during optimization" << std::endl;
	}

	for (auto i = 0; i < mat_size; ++i)
	{
		delete[] A[i];
	}
	delete[] A;

	std::cout << "Finished, press any key to leave...";
	std::getchar();

	return 0;
}


