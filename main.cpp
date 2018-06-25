#include <iostream>
#include <armadillo>
#include <chrono>
#include <random>
#include <numeric>
#include <unordered_map>
#include <gurobi_c++.h>

using std::vector;
using std::unordered_map;
using std::string;
using std::cout;
using std::endl;

using arma::s32_vec;
using arma::s32_mat;
using arma::mat;
using arma::uword;


/* Configuration */
constexpr bool SHOW_GUROBI_OUTPUT = true;
constexpr bool SHOW_MATRIX_OUTPUT = false;
constexpr uint32_t TIMEOUT_MATRIX_REDUCE = 10;

/* Parameters */
constexpr int q = 11;
constexpr int k = 3;
constexpr int b = 10;

s32_vec encode(const s32_vec& word, const s32_mat &G, int q);

int recursionHelper(vector<s32_vec> &rVector, int vPosition, int hPosition, int q, int k);
int getMinHammingDistance(int q, int k, const s32_mat &G);

s32_vec generateVectorX(const s32_mat &A, const s32_vec& c, int b);
s32_vec generateVectorC(const vector<vector<uword>>& colGroups);

s32_mat generateMatrixA(const vector<s32_vec>& canonicalRepresents, int q);
s32_mat generateMatrixG(const s32_vec& xVec, const vector<vector<uword>>& colGroups, const vector<s32_vec> &rVector);
s32_mat generateMatrixE0(int k, int q);

vector<s32_vec> generateCanonicalRepresent(int q, int k);
vector<s32_vec> generateLanguage(int q, int k);

void regenerateMatrixA(s32_mat& A, const unordered_map<uword, vector<uword>>& colCycles, const unordered_map<uword, vector<uword>>& rowCycles);
vector<vector<uword>> generateRepresentGroups(const unordered_map<uword, vector<uword>>& cycle, int vecCount);

void normalizeVector(s32_vec& v, int q);
bool isEqualVector(const s32_vec& v1, const s32_vec& v2);
bool isLinDependent(const s32_vec& v1, const s32_vec& v2, int q);
bool isNullVector(const s32_vec& rVector);

int getVectorIndex(const vector<s32_vec>& rVector, const s32_vec& rVec, int q);
unordered_map<uword, vector<uword>> cycleCheck(const vector<s32_vec>& rVector, const s32_mat& E, int q);


int main() 
{
	cout << "###### Linear Code Creator ######" << endl << endl;
    cout << "q = " << q << endl << "k = " << k << endl << "b = " << b << endl;

    cout << "Generate canonical represents" << endl;
    cout << "----------------------------------------------------" << endl;
	auto rVector = generateCanonicalRepresent(q, k);
	if (SHOW_MATRIX_OUTPUT)
	{
		for (uword i = 0; i < rVector.size(); ++i)
		{
			cout << "r" << i << "\t";
			rVector.at(i).t().raw_print();
		}
	}
    cout << "----------------------------------------------------" << endl << endl;

	cout << "Generate matrix A" << endl;
	cout << "----------------------------------------------------" << endl;
	auto A = generateMatrixA(rVector, q);
	if (SHOW_MATRIX_OUTPUT)
	{
		A.raw_print();
	}
	s32_vec c(A.n_cols);
	c.fill(1);
	cout << "----------------------------------------------------" << endl << endl;

	cout << "Testing for e0..." << endl;
	cout << "----------------------------------------------------" << endl;

	unordered_map<uword, vector<uword>> colCycles;
	vector<vector<uword>> colGroups;

	s32_mat e0(k, k);
	s32_mat test_e0(k, k);
	test_e0.fill(0);
	auto colGroupCount = std::numeric_limits<uint32_t>::max();
	uint32_t timeout = 0;

	std::chrono::high_resolution_clock::time_point start;
	do
	{
		colCycles = cycleCheck(rVector, test_e0, q);
		colGroups = generateRepresentGroups(colCycles, rVector.size());

		uint32_t maxElements = 0;
		for (auto& colGroup : colGroups)
		{
			if (colGroup.size() > maxElements)
			{
				maxElements = colGroup.size();
			}
		}
		if (maxElements < q && colGroups.size() < colGroupCount)
		{
			colGroupCount = colGroups.size();
			e0 = test_e0;
			timeout = 0;
		}

		const auto end = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		if (duration > 1000)
		{
			cout << "Current producer e0 reduces A from [" << A.n_cols << "] to [" << colGroupCount << "] columns." << endl;
			start = end;
			++timeout;
		}

		test_e0 = generateMatrixE0(k, q);
	} while (colGroupCount == A.n_cols || timeout < TIMEOUT_MATRIX_REDUCE);

	colCycles = cycleCheck(rVector, e0, q);
	colGroups = generateRepresentGroups(colCycles, rVector.size());

	const s32_mat e0_t = e0.t();
	const auto rowCycles = cycleCheck(rVector, e0_t, q);
	auto rowGroups = generateRepresentGroups(rowCycles, rVector.size());

	cout << "Found Producer e0 - reducing to [" << colGroups.size() << "] groups" << endl;
	e0.raw_print();
	cout << "----------------------------------------------------" << endl << endl;

	cout << "Found cycles for columns" << endl;
	cout << "----------------------------------------------------" << endl;
	for (auto& group : colGroups)
	{
		cout << "S([r" << group.at(0) << "]) = {";
		for (uword i = 0; i < group.size(); i++)
		{
			cout << "[r" << group.at(i) << "]";
			if (i != group.size() - 1) cout << ", ";
		}
		cout << "}" << endl;
	}

	cout << endl;
	cout << "Found cycles for rows" << endl;
	cout << "----------------------------------------------------" << endl;
	for (auto& group : rowGroups)
	{
		cout << "ST([r" << group.at(0) << "]) = {";
		for (uword i = 0; i < group.size(); i++)
		{
			cout << "[r" << group.at(i) << "]";
			if (i != group.size() - 1) cout << ", ";
		}
		cout << "}" << endl;
	}
	cout << "----------------------------------------------------" << endl << endl;

	cout << "Re-calculate matrix A with cylces" << endl;
	cout << "----------------------------------------------------" << endl;
	regenerateMatrixA(A, colCycles, rowCycles);
	if (SHOW_MATRIX_OUTPUT)
	{
		A.raw_print();
	}
	cout << "----------------------------------------------------" << endl << endl;

	cout << "Calculate vector c" << endl;
	cout << "----------------------------------------------------" << endl;
	c = generateVectorC(colGroups);
	c.t().raw_print();
	cout << "----------------------------------------------------" << endl << endl;

	auto x = generateVectorX(A, c, b);
    cout << "Calculating vector x" << endl;
    cout << "----------------------------------------------------" << endl;
	x.t().raw_print();
    cout << "----------------------------------------------------" << endl << endl;

    cout << "We have now following code" << endl;
    cout << "----------------------------------------------------" << endl;
	const auto n = dot(x, c);
	const auto d = n - b;
    cout << "[n,k,d]_q" << endl;
    cout << "[" << n << "," << k << "," << d << "]_" << q << "" << endl;
    cout << "----------------------------------------------------" << endl << endl;

    cout << "Generate Generatormatrix G" << endl;
    cout << "----------------------------------------------------" << endl;
	auto G = generateMatrixG(x, colGroups, rVector);
    G.raw_print();
    cout << "----------------------------------------------------" << endl << endl;

    cout << "Proof Hamming Distance" << endl;
    cout << "----------------------------------------------------" << endl;
	const int minDistance = getMinHammingDistance(q, k, G);
    cout << minDistance << " = calculated distance" << endl;
    cout << d << " = should be the distance" << endl;
    cout << "----------------------------------------------------" << endl << endl;

	cout << "Press any key to exit...";
	getchar();

    return 0;
}

template<class T, class I1, class I2>
T hamming_distance(uint32_t size, I1 b1, I2 b2) 
{
	return std::inner_product(b1, b1 + size, b2, T{},
		std::plus<T>(), std::not_equal_to<T>());
}

int getMinHammingDistance(const int q, const int k, const s32_mat &G)
{
	auto language = generateLanguage(q, k);
	language.erase(language.begin());

    vector<s32_vec> codes;
	codes.reserve(language.size());
	for (const auto& word : language)
    {
        codes.push_back(encode(word, G, q));
    }

	auto minDistance = std::numeric_limits<int>::max();
	for (auto& codeword : codes)
	{
		for (auto& codeword2 : codes)
		{
			const auto distance = hamming_distance<int>(codeword.size(), codeword.begin(), codeword2.begin());
			if (distance > 0 && distance < minDistance) 
			{
				minDistance = distance;
			}
		}
	}

    return minDistance;
}

s32_vec encode(const s32_vec& word, const s32_mat &G, const int q) 
{
	s32_vec encoded = (word.t() * G).t();
	normalizeVector(encoded, q);

	return encoded;
}

vector<s32_vec> generateLanguage(const int q, const int k)
{
	auto rVector = generateCanonicalRepresent(q, k + 1);

    vector<s32_vec> language;
    for (auto current : rVector)
    {
	    s32_vec newtmp(k);
        if (current.at(0) == 1) {
            for (auto j = 1; j < k + 1; j++) {
                newtmp[j - 1] = current[j];
            }
            language.push_back(newtmp);
        }
    }

    return language;
}

s32_mat generateMatrixG(const s32_vec& xVec, const vector<vector<uword>>& colGroups, const vector<s32_vec> &rVector)
{
	uint32_t colSize = 0;
	for (uint32_t i = 0; i < xVec.size(); ++i)
	{
		colSize += xVec.at(i) * colGroups.at(i).size();
	}

	const auto rowSize = rVector.at(0).size();
	s32_mat G(rowSize, colSize, arma::fill::zeros);

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

s32_mat generateMatrixE0(const int k, const int q)
{
	std::default_random_engine generator;
	const std::uniform_int_distribution<int> distribution(0, q - 1);

	mat e(k, k);
	do
	{
		for (auto i = 0; i < k; ++i)
		{
			for (auto j = 0; j < k; ++j)
			{
				generator.seed(std::random_device()());
				e(i, j) = distribution(generator);
			}
		}
	} while (0 == det(e));

	s32_mat e0(k, k);
	for (auto i = 0; i < k; ++i)
	{
		for (auto j = 0; j < k; ++j)
		{
			e0(i, j) = e(i, j);
		}
	}

	return e0;
}

void normalizeVector(s32_vec& v, const int q)
{
	for (auto& value : v)
	{
		value %= q;
	}
}

bool isEqualVector(const s32_vec& v1, const s32_vec& v2)
{
	bool ret = false;

	if (v1.size() == v2.size())
	{
		ret = true;
		for (uword i = 0; i < v1.size(); ++i)
		{
			if (v1.at(i) != v2.at(i))
			{
				ret = false;
			}
		}
	}

	return ret;
}

bool isLinDependent(const s32_vec& v1, const s32_vec& v2, const int q)
{
	bool ret = false;

	if (v1.size() == v2.size())
	{
		for (auto i = 1; i < q; ++i)
		{
			s32_vec tmp = v2 * i;
			normalizeVector(tmp, q);
			if (isEqualVector(v1, tmp))
			{
				ret = true;
				break;
			}
		}
	}

	return ret;
}

bool isNullVector(const s32_vec& rVector)
{
	bool ret = true;

	for (auto& r : rVector)
	{
		if (0 != r)
		{
			ret = false;
		}
	}

	return ret;
}

int getVectorIndex(const vector<s32_vec>& rVector, const s32_vec& rVec, const int q)
{
	auto ret = -1;

	for (uword i = 0; i < rVector.size(); i++)
	{
		if (isLinDependent(rVector.at(i), rVec, q))
		{
			ret = i;
			break;
		}
	}

	return ret;
}

unordered_map<uword, vector<uword>> cycleCheck(const vector<s32_vec>& rVector, const s32_mat& E, const int q)
{
	unordered_map<uword, uword> convert;
	for (uword i = 0; i < rVector.size(); i++)
	{
		s32_vec tmpVec = E * rVector.at(i);
		normalizeVector(tmpVec, q);
		if (!isLinDependent(rVector.at(i), tmpVec, q))
		{
			const auto vectorIndex = getVectorIndex(rVector, tmpVec, q);
			if (-1 != vectorIndex)
			{
				//cout << "Found conversion: r" << i << " -> r" << vectorIndex << endl;
				convert.insert({ i, vectorIndex });
			}
		}
	}

	unordered_map<uword, vector<uword>> cycles;
	for (uword i = 0; i < rVector.size(); i++)
	{
		const auto key = convert.find(i);
		if (key != convert.end())
		{
			vector<uword> tmp;
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
				sort(tmp.begin(), tmp.end());
				if (cycles.find(tmp.front()) == cycles.end())
				{
					cycles.insert({ tmp.front(), tmp });
				}
			}
		}
	}

	return cycles;
}

s32_vec generateVectorX(const s32_mat &A, const s32_vec& c, const int b)
{
    try {
	    const auto env = GRBEnv();
	    auto model = GRBModel(env);
		model.getEnv().set(GRB_IntParam_OutputFlag, SHOW_GUROBI_OUTPUT);

        vector<GRBVar> modelVars;
	    modelVars.reserve(c.size());
	    for (uint32_t i = 0; i < c.size(); i++)
		{
            modelVars.push_back(model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER));
        }

        GRBLinExpr expr = 0.0;
        for (uint32_t i = 0; i < c.size(); i++)
		{
            expr += modelVars.at(i) * c.at(i);
        }

        model.setObjective(expr, GRB_MAXIMIZE);

        for (uword x = 0; x < A.n_rows; x++) 
		{
            expr = 0.0;
            for (uword y = 0; y < A.n_cols; y++) 
			{
                expr += modelVars.at(y) * A(x, y);
            }
            model.addConstr(expr, GRB_LESS_EQUAL, b);
        }

		//model.update();
		//model.write("debug.lp");
        model.optimize();

		s32_vec x(modelVars.size());
		for (uint32_t i = 0; i < modelVars.size(); ++i)
		{
			x.at(i) = static_cast<int>(modelVars.at(i).get(GRB_DoubleAttr_X));
		}

        return x;
    } catch (const GRBException& e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
    } catch (...) {
        cout << "Exception during optimization" << endl;
    }

    return s32_vec(A.n_cols);
}

s32_vec generateVectorC(const vector<vector<uword>>& colGroups)
{
	s32_vec c(colGroups.size());

	for (uword i = 0; i < colGroups.size(); ++i)
	{
		c.at(i) = colGroups.at(i).size();
	}

	return c;
}

s32_mat generateMatrixA(const vector<s32_vec> &canonicalRepresents, const int q) 
{
	s32_mat A(canonicalRepresents.size(), canonicalRepresents.size(), arma::fill::zeros);

    for (uword row = 0; row < canonicalRepresents.size(); ++row) 
	{
        for (uword col = 0; col < canonicalRepresents.size(); ++col)
		{
            if (static_cast<int>(dot(canonicalRepresents.at(row), canonicalRepresents.at(col))) % q == 0) 
			{
                A(row, col) = 1;
            }
        }
    }
    return A;
}

void regenerateMatrixA(s32_mat& A, const unordered_map<uword, vector<uword>>& colCycles, const unordered_map<uword, vector<uword>>& rowCycles)
{
	vector<uword> colsDelete;
	vector<uword> rowsDelete;

	for (auto& c : colCycles)
	{
		for (uword i = 1; i < c.second.size(); ++i)
		{
			for (uword j = 0; j < A.n_rows; j++)
			{
				A(j, c.second.at(0)) += A(j, c.second.at(i));
			}

			colsDelete.push_back(c.second.at(i));
		}
	}

	for (auto& r : rowCycles)
	{
		for (uword i = 1; i < r.second.size(); ++i)
		{
			rowsDelete.push_back(r.second.at(i));
		}
	}

	sort(colsDelete.begin(), colsDelete.end());
	reverse(colsDelete.begin(), colsDelete.end());
	for (auto& c : colsDelete)
	{
		A.shed_col(c);
	}

	sort(rowsDelete.begin(), rowsDelete.end());
	reverse(rowsDelete.begin(), rowsDelete.end());
	for (auto& r : rowsDelete)
	{
		A.shed_row(r);
	}
}

vector<s32_vec> generateCanonicalRepresent(const int q, const int k)
{
    vector<s32_vec> rVector;
	auto tmp = s32_vec(k);
    tmp.fill(0);
    rVector.push_back(tmp);
    rVector.at(0).at(k - 1) = 1;

    for (auto i = k - 2; i >= 0; i--) 
	{
		s32_vec tmp2(k);
        for (int j = 0; j < k; j++) {
			tmp2.at(j) = 0;
        }
		tmp2.at(i) = 1;
        rVector.push_back(tmp2);

        recursionHelper(rVector, rVector.size() - 1, i + 1, q, k);
    }

    return rVector;
}

bool sortHelper(const vector<uword>& i, const vector<uword>& j)
{
	return i.at(0) < j.at(0);
}

vector<vector<uword>> generateRepresentGroups(const unordered_map<uword, vector<uword>>& cycle, const int vecCount)
{
	vector<vector<uword>> repGroups;
	repGroups.reserve(cycle.size());
	for (auto& c : cycle)
	{
		repGroups.push_back(c.second);
	}

	for (auto i = 0; i < vecCount; ++i)
	{
		bool found = false;
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
			repGroups.push_back(std::vector<uword>{static_cast<uword>(i)});
		}
	}

	std::sort(repGroups.begin(), repGroups.end(), sortHelper);

	return repGroups;
}

int recursionHelper(vector<s32_vec> &rVector, int vPosition, int hPosition, const int q, const int k) 
{
	auto tmp = rVector.at(vPosition);
    rVector.pop_back();
    for (int i = 0; i < q; i++) {
        tmp.at(hPosition) = i;
        rVector.push_back(tmp);
        vPosition++;
        if (hPosition < k - 1) {
	        const int newVPosition = recursionHelper(rVector, vPosition - 1, hPosition + 1, q, k);
            vPosition = newVPosition;
        }
    }
    return vPosition;
}
