#include <iostream>
#include <armadillo>
#include <chrono>
#include <gurobi_c++.h>

using std::vector;
using arma::s32_vec;
using std::cout;
using std::endl;
using arma::s32_mat;
using std::string;
using std::to_string;
using std::numeric_limits;

constexpr bool SHOW_GUROBI_OUTPUT = true;
constexpr bool SHOW_MATRIX_OUTPUT = true;
constexpr uint32_t TIMEOUT_MATRIX_REDUCE = 10;

constexpr int q = 3;
constexpr int k = 5;
constexpr int b = 6;

s32_vec encode(const s32_vec& word, const s32_mat &G, int q);

int recursionHelper(vector<s32_vec> &rVector, int vPosition, int hPosition, int q, int k);
int getMinHammingDistance(int q, int k, const s32_mat &G);
int getHammingDistance(const s32_vec& c1, const s32_vec& c2);

s32_mat generateMatrixA(const vector<s32_vec>& canonicalRepresents, int q);
s32_vec generateX(const s32_mat &A, const s32_vec& c, int b);
s32_vec generateC(const vector<vector<arma::uword>>& colGroups);
s32_mat generateG(const s32_vec& xVec, const vector<vector<arma::uword>>& colGroups, const vector<s32_vec> &rVector);
s32_mat generateE0(int k);

vector<s32_vec> generateCanonicalRepresent(int q, int k);
vector<s32_vec> generateLanguage(int q, int k);

void regenerateMatrixA(s32_mat& A, const vector<vector<arma::uword>>& colCycles, const vector<vector<arma::uword>>& rowCycles);
vector<vector<arma::uword>> generateRepresentGroups(const vector<vector<arma::uword>>& colCycles, int vecCount);

void normalizeVector(s32_vec& v, int q);
bool isEqualVector(const s32_vec& v1, const s32_vec& v2);
bool isLinDependent(const s32_vec& v1, const s32_vec& v2, int q);
bool isNullVector(const s32_vec& rVector);

int getVectorIndex(const vector<s32_vec>& rVector, const s32_vec& rVec, int q);
vector<vector<arma::uword>> cycleCheck(const vector<s32_vec>& rVector, const s32_mat& E, int q);



int main() 
{
	cout << "###### Linear Code Creator ######" << endl << endl;
    cout << "q = " << q << endl << "k = " << k << endl << "b = " << b << endl;

    cout << "Generate canonical represents" << endl;
    cout << "----------------------------------------------------" << endl;
	auto rVector = generateCanonicalRepresent(q, k);
	if (SHOW_MATRIX_OUTPUT)
	{
		for (arma::uword i = 0; i < rVector.size(); ++i)
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

	vector<vector<arma::uword>> colCycles;
	vector<vector<arma::uword>> colGroups;
	vector<vector<arma::uword>> rowCycles;
	vector<vector<arma::uword>> rowGroups;

	s32_mat e0;
	auto totalGroupCount = numeric_limits<uint32_t>::max();
	auto colGroupsCount = numeric_limits<uint32_t>::max();
	auto rowGroupsCount = numeric_limits<uint32_t>::max();
	uint32_t timeout = 0;

	auto start = std::chrono::high_resolution_clock::now();
	do
	{
		auto test_e0 = generateE0(k);
		colCycles = cycleCheck(rVector, test_e0, q);
		colGroups = generateRepresentGroups(colCycles, rVector.size());
		const s32_mat test_e0_t = test_e0.t();
		rowCycles = cycleCheck(rVector, test_e0_t, q);
		rowGroups = generateRepresentGroups(rowCycles, rVector.size());

		if (colGroups.size() + rowGroups.size() < totalGroupCount)
		{
			colGroupsCount = colGroups.size();
			rowGroupsCount = rowGroups.size();
			totalGroupCount = colGroupsCount + rowGroupsCount;
			e0 = test_e0;
			timeout = 0;
		}

		const auto end = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		if (duration > 1000)
		{
			cout << "Current producer e0 creates " << colGroupsCount << " columns and " << rowGroupsCount << " rows." << endl;
			start = end;
			++timeout;
		}
	} while (timeout < TIMEOUT_MATRIX_REDUCE);

	colCycles = cycleCheck(rVector, e0, q);
	colGroups = generateRepresentGroups(colCycles, rVector.size());

	const s32_mat e0_t = e0.t();
	rowCycles = cycleCheck(rVector, e0_t, q);
	rowGroups = generateRepresentGroups(rowCycles, rVector.size());

	cout << "Found Producer e0 - reducing to [" << colGroups.size() << "] groups" << endl;
	e0.raw_print();
	cout << "----------------------------------------------------" << endl << endl;

	cout << "Found cycles for columns" << endl;
	cout << "----------------------------------------------------" << endl;
	for (auto& group : colGroups)
	{
		cout << "S([r" << group.at(0) << "]) = {";
		for (arma::uword i = 0; i < group.size(); i++)
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
		for (arma::uword i = 0; i < group.size(); i++)
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
	c = generateC(colGroups);
	c.t().raw_print();
	cout << "----------------------------------------------------" << endl << endl;

	auto x = generateX(A, c, b);
    cout << "Calculating vector x" << endl;
    cout << "----------------------------------------------------" << endl;
	x.t().raw_print();
    cout << "----------------------------------------------------" << endl << endl;

    cout << "We have now following code" << endl;
    cout << "----------------------------------------------------" << endl;
	uint32_t n = 0;
	for (uint32_t i = 0; i < x.size(); ++i)
	{
		n += x.at(i) * c.at(i);
	}

	const int d = n - b;
    cout << "[n,k,d]_q" << endl;
    cout << "[" << n << "," << k << "," << d << "]_" << q << "" << endl;
    cout << "----------------------------------------------------" << endl << endl;

    cout << "Generate Generatormatrix G" << endl;
    cout << "----------------------------------------------------" << endl;
	auto G = generateG(x, colGroups, rVector);
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

int getMinHammingDistance(const int q, const int k, const s32_mat &G)
{
	auto language = generateLanguage(q, k);

    vector<s32_vec> codes;
	codes.reserve(language.size());
	for (const auto& word : language)
    {
        codes.push_back(encode(word, G, q));
    }

	auto minDistance = numeric_limits<int>::max();
	for (auto& codeword : codes)
	{
		for (auto& codeword2 : codes)
		{
			const auto distance = getHammingDistance(codeword, codeword2);
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

int getHammingDistance(const s32_vec& c1, const s32_vec& c2) 
{
    if (c1.size() != c2.size())
        return -1;

    int ret = 0;

    for (arma::uword i = 0; i < c1.size(); i++) 
	{
		ret += abs(c1.at(i) - c2.at(i));
    }

    return ret;
}

vector<s32_vec> generateLanguage(const int q, const int k)
{
	auto rVector = generateCanonicalRepresent(q, k + 1);

    vector<s32_vec> language;
    for (auto current : rVector)
    {
	    s32_vec newtmp(k);
        if (current.at(0) == 1) {
            for (int j = 1; j < k + 1; j++) {
                newtmp[j - 1] = current[j];
            }
            language.push_back(newtmp);
        }
    }

    return language;
}

s32_mat generateG(const s32_vec& xVec, const vector<vector<arma::uword>>& colGroups, const vector<s32_vec> &rVector)
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

s32_mat generateE0(const int k)
{
	arma::arma_rng::set_seed_random();
	s32_mat e = arma::randn<s32_mat>(k, k);

	for (auto i = 0; i < k; ++i)
	{
		for (auto j = 0; j < k; ++j)
		{
			e(i, j) = e(i, j) != 0 ? 1 : 0;
		}
	}

	return e;
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
		for (arma::uword i = 0; i < v1.size(); ++i)
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

	for (arma::uword i = 0; i < rVector.size(); i++)
	{
		if (isLinDependent(rVector.at(i), rVec, q))
		{
			ret = i;
			break;
		}
	}

	return ret;
}

vector<vector<arma::uword>> cycleCheck(const vector<s32_vec>& rVector, const s32_mat& E, const int q)
{
	vector<vector<arma::uword>> cycles;

	for (arma::uword i = 0; i < rVector.size(); i++)
	{
		s32_vec tmpVec = E * rVector.at(i);
		normalizeVector(tmpVec, q);
		if (!isLinDependent(rVector.at(i), tmpVec, q))
		{
			const auto vectorIndex = getVectorIndex(rVector, tmpVec, q);
			if (-1 != vectorIndex)
			{
				//cout << "Found conversion: r" << i << " -> r" << vectorIndex << endl;
				auto isInQueue = false;
				for (auto& c : cycles)
				{
					if (c.back() == i)
					{
						c.push_back(vectorIndex);
						isInQueue = true;
					}
					else if (c.front() == static_cast<arma::uword>(vectorIndex))
					{
						vector<arma::uword> r;
						r.push_back(i);
						r.insert(r.end(), c.begin(), c.end());
						c = r;
						isInQueue = true;
					}
				}

				if (!isInQueue)
				{
					vector<arma::uword> tmpQueue;
					tmpQueue.push_back(i);
					tmpQueue.push_back(vectorIndex);
					cycles.push_back(tmpQueue);
				}
			}
		}
	}

	for (auto& c : cycles)
	{
		if (c.front() != c.back())
		{
			s32_vec tmpVec = E * rVector.at(c.back());
			normalizeVector(tmpVec, q);
			const arma::uword vectorIndex = getVectorIndex(rVector, tmpVec, q);
			if (c.front() == vectorIndex)
			{
				c.push_back(vectorIndex);
			}
		}
	}

	for (auto it = cycles.begin(); it != cycles.end(); )
	{
		if ((*it).front() != (*it).back())
		{
			cycles.erase(it);
		}
		else
		{
			(*it).pop_back();
			std::sort((*it).begin(), (*it).end());
			++it;
		}
	}

	return cycles;
}

s32_vec generateX(const s32_mat &A, const s32_vec& c, const int b)
{
    try {
	    const auto env = GRBEnv();
	    auto model = GRBModel(env);
		model.getEnv().set(GRB_IntParam_OutputFlag, SHOW_GUROBI_OUTPUT);

        vector<GRBVar> modelVars;

	    modelVars.reserve(A.n_cols);
	    for (arma::uword i = 0; i < A.n_cols; i++) {
            modelVars.push_back(model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER, string("var").append(to_string(i))));
        }

        GRBLinExpr expr = 0.0;
        for (arma::uword i = 0; i < A.n_cols; i++) {
            expr += modelVars.at(i) * c.at(i);
        }

        model.setObjective(expr, GRB_MAXIMIZE);

        for (arma::uword x = 0; x < A.n_rows; x++) {
            expr = 0.0;
            for (arma::uword y = 0; y < A.n_cols; y++) {
                expr += modelVars.at(y) * A(x, y);
            }
            model.addConstr(expr, GRB_LESS_EQUAL, b, string("constr").append(to_string(x)));
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

s32_vec generateC(const vector<vector<arma::uword>>& colGroups)
{
	s32_vec c(colGroups.size());

	for (arma::uword i = 0; i < colGroups.size(); ++i)
	{
		c.at(i) = colGroups.at(i).size();
	}

	return c;
}

s32_mat generateMatrixA(const vector<s32_vec> &canonicalRepresents, const int q) 
{
	s32_mat A(canonicalRepresents.size(), canonicalRepresents.size(), arma::fill::zeros);

    for (arma::uword row = 0; row < canonicalRepresents.size(); ++row) 
	{
        for (arma::uword col = 0; col < canonicalRepresents.size(); ++col)
		{
            if (static_cast<int>(dot(canonicalRepresents.at(row), canonicalRepresents.at(col))) % q == 0) 
			{
                A(row, col) = 1;
            }
        }
    }
    return A;
}

void regenerateMatrixA(s32_mat& A, const vector<vector<arma::uword>>& colCycles, const vector<vector<arma::uword>>& rowCycles)
{
	vector<arma::uword> colsDelete;
	vector<arma::uword> rowsDelete;

	for (auto& c : colCycles)
	{
		for (arma::uword i = 1; i < c.size(); ++i)
		{
			for (arma::uword j = 0; j < A.n_rows; j++)
			{
				A(j, c.at(0)) += A(j, c.at(i));
			}

			colsDelete.push_back(c.at(i));
		}
	}

	for (auto& r : rowCycles)
	{
		for (arma::uword i = 1; i < r.size(); ++i)
		{
			rowsDelete.push_back(r.at(i));
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

bool sortHelper(const vector<arma::uword>& i, const vector<arma::uword>& j)
{
	return i.at(0) < j.at(0);
}

vector<vector<arma::uword>> generateRepresentGroups(const vector<vector<arma::uword>>& colCycles, const int vecCount)
{
	vector<vector<arma::uword>> repGroups;
	repGroups.reserve(colCycles.size());
	for (auto& colCycle : colCycles)
	{
		repGroups.push_back(colCycle);
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
			repGroups.push_back(std::vector<arma::uword>{static_cast<arma::uword>(i)});
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
