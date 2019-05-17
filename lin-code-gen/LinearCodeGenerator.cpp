#include "LinearCodeGenerator.h"
#include "Math.h"

#include <unordered_map>
#include <gurobi_c++.h>
#include <armadillo>
#include <random>

/* Configuration */
constexpr bool SHOW_GUROBI_OUTPUT = true;
constexpr bool SHOW_MATRIX_OUTPUT = false;
constexpr uint32_t TIMEOUT_MATRIX_REDUCE = 1;

LinearCodeGenerator::LinearCodeGenerator(uint16_t q, uint16_t k, uint16_t b)
	: m_q(q)
	, m_k(k)
	, m_b(b)
{
	
}

auto LinearCodeGenerator::generateCode() const -> LinearCode
{
	std::cout << "Generating canonical represents" << std::endl;
	std::cout << "----------------------------------------------------" << std::endl;

	auto canonialRepresents = Math::generateCanonialRepresents(m_q, m_k);
	if (SHOW_MATRIX_OUTPUT)
	{
		for (arma::uword i = 0; i < canonialRepresents.size(); ++i)
		{
			std::cout << "r" << i << "\t";
			canonialRepresents.at(i).t().raw_print();
		}
	}
	std::cout << "----------------------------------------------------" << std::endl << std::endl;



	std::cout << "Generating matrix A" << std::endl;
	std::cout << "----------------------------------------------------" << std::endl;
	const auto matrixA = generateMatrixA(canonialRepresents);
	if (SHOW_MATRIX_OUTPUT)
	{
		matrixA.raw_print();
	}
	std::cout << "----------------------------------------------------" << std::endl << std::endl;



	std::cout << "Generating producer e0" << std::endl;
	std::cout << "----------------------------------------------------" << std::endl;
	const auto producerE0 = generateProducerE0(canonialRepresents, matrixA);

	const auto colCycles = cycleCheck(canonialRepresents, producerE0, m_q);
	const auto rowCycles = cycleCheck(canonialRepresents, producerE0.t(), m_q);

	auto colGroups = generateRepresentGroups(colCycles, canonialRepresents.size());
	auto rowGroups = generateRepresentGroups(rowCycles, canonialRepresents.size());

	std::cout << "Found Producer e0 - reducing to [" << colGroups.size() << "] groups" << std::endl;
	if (SHOW_MATRIX_OUTPUT)
	{
		producerE0.raw_print();
		std::cout << "----------------------------------------------------" << std::endl << std::endl;
		std::cout << "Found cycles for columns" << std::endl;
		std::cout << "----------------------------------------------------" << std::endl;
		for (auto& group : colGroups)
		{
			std::cout << "S([r" << group.at(0) << "]) = {";
			for (arma::uword i = 0; i < group.size(); i++)
			{
				std::cout << "[r" << group.at(i) << "]";
				if (i != group.size() - 1) std::cout << ", ";
			}
			std::cout << "}" << std::endl;
		}

		std::cout << std::endl;
		std::cout << "Found cycles for rows" << std::endl;
		std::cout << "----------------------------------------------------" << std::endl;
		for (auto& group : rowGroups)
		{
			std::cout << "ST([r" << group.at(0) << "]) = {";
			for (arma::uword i = 0; i < group.size(); i++)
			{
				std::cout << "[r" << group.at(i) << "]";
				if (i != group.size() - 1) std::cout << ", ";
			}
			std::cout << "}" << std::endl;
		}
	}
	std::cout << "----------------------------------------------------" << std::endl << std::endl;



	std::cout << "Re-calculate matrix A with cylces" << std::endl;
	std::cout << "----------------------------------------------------" << std::endl;
	const auto optMatrixA = generateOptimizedMatrixA(matrixA, colCycles, rowCycles);
	if (SHOW_MATRIX_OUTPUT)
	{
		optMatrixA.raw_print();
	}
	std::cout << "----------------------------------------------------" << std::endl << std::endl;



	std::cout << "Calculate vector c" << std::endl;
	std::cout << "----------------------------------------------------" << std::endl;
	const auto vectorC = generateVectorC(colGroups);
	vectorC.t().raw_print();
	std::cout << "----------------------------------------------------" << std::endl << std::endl;



	std::cout << "Calculating vector x" << std::endl;
	std::cout << "----------------------------------------------------" << std::endl;
	auto vectorX = generateVectorX(optMatrixA, vectorC, m_b);
	vectorX.t().raw_print();
	std::cout << "----------------------------------------------------" << std::endl << std::endl;



	const uint16_t n = dot(vectorX, vectorC);
	const uint16_t d = n - m_b;
	auto G = generateMatrixG(vectorX, colGroups, canonialRepresents);

	return { n, m_q, m_k, d, G };
}


auto LinearCodeGenerator::generateProducerE0(const std::vector<arma::s32_vec>& canonialRepresents, const arma::s32_mat& matrixA) const -> arma::s32_mat
{
	arma::s32_mat e0(m_k, m_k);
	arma::s32_mat test_e0(m_k, m_k);
	test_e0.fill(0);
	auto colGroupCount = std::numeric_limits<uint32_t>::max();
	uint32_t timeout = 0;

	std::chrono::high_resolution_clock::time_point start;
	do
	{
		auto colCycles = cycleCheck(canonialRepresents, test_e0, m_q);
		auto colGroups = generateRepresentGroups(colCycles, canonialRepresents.size());

		uint32_t maxElements = 0;
		for (auto& colGroup : colGroups)
		{
			if (colGroup.size() > maxElements)
			{
				maxElements = colGroup.size();
			}
		}
		if (maxElements < m_q && colGroups.size() < colGroupCount)
		{
			colGroupCount = colGroups.size();
			e0 = test_e0;
			timeout = 0;
		}

		const auto end = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		if (duration > 1000)
		{
			std::cout << "Current producer e0 reduces A from [" << matrixA.n_cols << "] to [" << colGroupCount << "] columns." << std::endl;
			start = end;
			++timeout;
		}

		test_e0 = generateMatrixE0();
	} while (colGroupCount == matrixA.n_cols || timeout < TIMEOUT_MATRIX_REDUCE);

	return e0;
}

auto LinearCodeGenerator::generateOptimizedMatrixA(const arma::s32_mat& matrixA,
	const std::unordered_map<arma::uword, std::vector<arma::uword>>& colCycles,
	const std::unordered_map<arma::uword, std::vector<arma::uword>>& rowCycles) const -> arma::s32_mat
{
	auto optMatrixA = matrixA;
	std::vector<arma::uword> colsDelete;
	std::vector<arma::uword> rowsDelete;

	for (auto& c : colCycles)
	{
		for (arma::uword i = 1; i < c.second.size(); ++i)
		{
			for (arma::uword j = 0; j < optMatrixA.n_rows; j++)
			{
				optMatrixA(j, c.second.at(0)) += optMatrixA(j, c.second.at(i));
			}

			colsDelete.push_back(c.second.at(i));
		}
	}

	for (auto& r : rowCycles)
	{
		for (arma::uword i = 1; i < r.second.size(); ++i)
		{
			rowsDelete.push_back(r.second.at(i));
		}
	}

	sort(colsDelete.begin(), colsDelete.end());
	reverse(colsDelete.begin(), colsDelete.end());
	for (auto& c : colsDelete)
	{
		optMatrixA.shed_col(c);
	}

	sort(rowsDelete.begin(), rowsDelete.end());
	reverse(rowsDelete.begin(), rowsDelete.end());
	for (auto& r : rowsDelete)
	{
		optMatrixA.shed_row(r);
	}

	return optMatrixA;
}

auto LinearCodeGenerator::generateMatrixG(const arma::s32_vec& xVec, const std::vector<std::vector<arma::uword>>& colGroups,
	const std::vector<arma::s32_vec>& rVector) -> arma::s32_mat
{
	uint32_t colSize = 0;
	for (uint32_t i = 0; i < xVec.size(); ++i)
	{
		colSize += xVec.at(i) * colGroups.at(i).size();
	}

	const auto rowSize = rVector.at(0).size();
	arma::s32_mat G(rowSize, colSize, arma::fill::zeros);

	int colIndex = 0;
	for (uint32_t x = 0; x < xVec.size(); ++x)
	{
		for (auto i = 0; i < xVec.at(x); ++i)
		{
			for (unsigned int j : colGroups.at(x))
			{
				for (uint32_t k = 0; k < rowSize; ++k)
				{
					G(k, colIndex) = rVector.at(j).at(k);
				}
				colIndex++;
			}
		}
	}

	return G;
}

auto LinearCodeGenerator::generateVectorC(const std::vector<std::vector<arma::uword>>& colGroups) -> arma::s32_vec
{
	arma::s32_vec vectorC(colGroups.size());

	for (arma::uword i = 0; i < colGroups.size(); ++i)
	{
		vectorC.at(i) = colGroups.at(i).size();
	}

	return vectorC;
}

auto LinearCodeGenerator::generateVectorX(const arma::s32_mat& matrixA, const arma::s32_vec& vectorC, uint16_t b) -> arma::s32_vec
{
	const auto env = GRBEnv();
	auto model = GRBModel(env);
	model.getEnv().set(GRB_IntParam_OutputFlag, SHOW_GUROBI_OUTPUT);

	std::vector<GRBVar> modelVars;
	modelVars.reserve(vectorC.size());
	for (uint32_t i = 0; i < vectorC.size(); i++)
	{
		modelVars.push_back(model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER));
	}

	GRBLinExpr expr = 0.0;
	for (uint32_t i = 0; i < vectorC.size(); i++)
	{
		expr += modelVars.at(i) * vectorC.at(i);
	}

	model.setObjective(expr, GRB_MAXIMIZE);

	for (arma::uword x = 0; x < matrixA.n_rows; x++)
	{
		expr = 0.0;
		for (arma::uword y = 0; y < matrixA.n_cols; y++)
		{
			expr += modelVars.at(y) * matrixA(x, y);
		}
		model.addConstr(expr, GRB_LESS_EQUAL, b);
	}

	//model.update();
	//model.write("debug.lp");
	model.optimize();

	arma::s32_vec vectorX(modelVars.size());
	for (uint32_t i = 0; i < modelVars.size(); ++i)
	{
		vectorX.at(i) = static_cast<int>(modelVars.at(i).get(GRB_DoubleAttr_X));
	}

	return vectorX;
}

auto LinearCodeGenerator::generateMatrixA(const std::vector<arma::s32_vec>& canonialRepresents) const -> arma::s32_mat
{
	auto matrixA = arma::s32_mat(canonialRepresents.size(), canonialRepresents.size(), arma::fill::zeros);

	for (arma::uword row = 0; row < canonialRepresents.size(); ++row)
	{
		for (arma::uword col = 0; col < canonialRepresents.size(); ++col)
		{
			if (static_cast<int>(dot(canonialRepresents.at(row), canonialRepresents.at(col))) % m_q == 0)
			{
				matrixA(row, col) = 1;
			}
		}
	}

	return matrixA;
}

auto sortHelper(const std::vector<arma::uword>& i, const std::vector<arma::uword>& j) -> bool
{
	return i.at(0) < j.at(0);
}

auto LinearCodeGenerator::generateRepresentGroups(const std::unordered_map<arma::uword, std::vector<arma::uword>>& cycle,
	int vecCount) const -> std::vector<std::vector<arma::uword>>
{
	std::vector<std::vector<arma::uword>> repGroups;
	repGroups.reserve(cycle.size());
	for (auto& c : cycle)
	{
		repGroups.push_back(c.second);
	}

	for (auto i = 0; i < vecCount; ++i)
	{
		auto found = false;
		for (auto& repGroup : repGroups)
		{
			for (int k : repGroup)
			{
				if (k == i)
				{
					found = true;
				}
			}
		}

		if (!found)
		{
			repGroups.push_back(std::vector<arma::uword>{static_cast<arma::uword>(i)});
		}
	}

	std::sort(repGroups.begin(), repGroups.end(), sortHelper);

	return repGroups;
}

auto LinearCodeGenerator::generateMatrixE0() const -> arma::s32_mat
{
	std::default_random_engine generator;
	const std::uniform_int_distribution<int> distribution(0, m_q - 1);

	arma::mat e(m_k, m_k);
	do
	{
		for (auto i = 0; i < m_k; ++i)
		{
			for (auto j = 0; j < m_k; ++j)
			{
				generator.seed(std::random_device()());
				e(i, j) = distribution(generator);
			}
		}
	} while (0 == arma::det(e));

	arma::s32_mat e0(m_k, m_k);
	for (auto i = 0; i < m_k; ++i)
	{
		for (auto j = 0; j < m_k; ++j)
		{
			e0(i, j) = e(i, j);
		}
	}

	return e0;
}

auto LinearCodeGenerator::cycleCheck(const std::vector<arma::s32_vec>& rVector, const arma::s32_mat& E, uint16_t q) const
-> std::unordered_map<arma::uword, std::vector<arma::uword>>
{
	std::unordered_map<arma::uword, arma::uword> convert;
	for (arma::uword i = 0; i < rVector.size(); i++)
	{
		arma::s32_vec tmpVec = E * rVector.at(i);
		Math::normalizeVector(tmpVec, q);
		if (!Math::isLinDependent(rVector.at(i), tmpVec, q))
		{
			const auto vectorIndex = Math::getVectorIndex(rVector, tmpVec, q);
			if (-1 != vectorIndex)
			{
				//cout << "Found conversion: r" << i << " -> r" << vectorIndex << endl;
				convert.insert({ i, vectorIndex });
			}
		}
	}

	std::unordered_map<arma::uword, std::vector<arma::uword>> cycles;
	for (arma::uword i = 0; i < rVector.size(); i++)
	{
		const auto key = convert.find(i);
		if (key != convert.end())
		{
			std::vector<arma::uword> tmp;
			tmp.push_back(key->first);
			tmp.push_back(key->second);

			bool search;
			do
			{
				search = false;
				const auto next = convert.find(tmp.back());
				if (next != convert.end())
				{
					const auto loop = find(tmp.begin() + 1, tmp.end(), next->second);
					if (loop != tmp.end())
					{
						tmp.erase(tmp.begin(), loop);
					}

					tmp.push_back(next->second);
					search = true;
				}
			} while (search && tmp.front() != tmp.back());

			if (tmp.front() == tmp.back())
			{
				tmp.pop_back();
				std::sort(tmp.begin(), tmp.end());
				if (cycles.find(tmp.front()) == cycles.end())
				{
					cycles.insert({ tmp.front(), tmp });
				}
			}
		}
	}

	return cycles;
}
