#include <iostream>
#include <armadillo>
#include <gurobi_c++.h>

using std::vector;
using arma::s32_vec;
using std::cout;
using std::endl;
using arma::s32_mat;
using std::string;
using std::to_string;
using std::numeric_limits;

#define SHOW_GUROBI_OUTPUT 1
#define OPTIMIZE_GROUPS 1
#define GROUP_REDUCE_FACTOR 0.4

const int q = 3;
const int k = 5;
const int b = 6;

vector<s32_vec> canonicalRepresent(int q, int k);

vector<vector<arma::uword>> generateRepresentGroups(vector<vector<arma::uword>>& colCycles, int vecCount);

int recursionHelper(vector<s32_vec> &rVector, int vPosition, int hPosition, int q, int k);

s32_mat generateMatrixA(const vector<s32_vec> &rVector, int q);

void regenerateMatrixA(s32_mat& A, vector<vector<arma::uword>>& colCycles, vector<vector<arma::uword>>& rowCycles);

vector<double> gurobiPart(const s32_mat &A, const s32_vec& c, int b);

s32_vec generateC(vector<vector<arma::uword>>& colGroups);

s32_mat generateG(vector<double> xVec, vector<vector<arma::uword>> colGroups, const vector<s32_vec> &rVector);

s32_mat generateE0(int k);

void normalizeVector(s32_vec& v, int q);

bool isEqualVector(s32_vec& v1, s32_vec& v2);

bool isLinDependent(s32_vec& v1, s32_vec v2, int q);

bool isNullVector(s32_vec& rVector);

int getVectorIndex(vector<s32_vec>& rVector, s32_vec& rVec, int q);

vector<vector<arma::uword>> cycleCheck(vector<s32_vec>& rVector, s32_mat& E, int q);

int getMinHammingDistance(int q, int k, const s32_mat &G);

vector<s32_vec> generateLanguage(int q, int k);

int getHammingDistance(const s32_vec &left, const s32_vec &right);

s32_vec encode(const s32_vec &row, const s32_mat &G, int q);

int main() {
    cout << "q = " << q << endl << "k = " << k << endl << "b = " << b << endl;
    cout << "####################################################" << endl << endl;

    cout << "Generate canonical represents" << endl;
    cout << "----------------------------------------------------" << endl;
	auto rVector = canonicalRepresent(q, k);
    for (arma::uword i = 0; i < rVector.size(); ++i)
    {
		cout << "r" << i << "\t";
		rVector.at(i).t().raw_print();
    }
    cout << "----------------------------------------------------" << endl << endl;

	cout << "Generate matrix A" << endl;
	cout << "----------------------------------------------------" << endl;
	auto A = generateMatrixA(rVector, q);
	A.raw_print();
	s32_vec c(A.n_cols);
	c.fill(1);
	cout << "----------------------------------------------------" << endl << endl;

	cout << "Testing for e0 which creates less or equal than [" << static_cast<int>(A.n_cols * GROUP_REDUCE_FACTOR) << "] groups" << endl;
	cout << "----------------------------------------------------" << endl;
	s32_mat e0;
	vector<vector<arma::uword>> colCycles;
	vector<vector<arma::uword>> colGroups;
	do
	{
		e0 = generateE0(k);
		colCycles = cycleCheck(rVector, e0, q);
		colGroups = generateRepresentGroups(colCycles, rVector.size());
	} while (OPTIMIZE_GROUPS && colGroups.size() > A.n_cols * GROUP_REDUCE_FACTOR);

	cout << "Found Producer e0 - reducing to [" << colGroups.size() << "] groups" << endl;
	e0.raw_print();
	cout << "----------------------------------------------------" << endl << endl;

	cout << "Found cycles" << endl;
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

	cout << endl << "Transposing e0..." << endl;
	e0 = e0.t();
	auto rowCycles = cycleCheck(rVector, e0, q);
	auto rowGroups = generateRepresentGroups(rowCycles, rVector.size());
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
	A.raw_print();
	cout << "----------------------------------------------------" << endl << endl;

	cout << "Calculate vector c" << endl;
	cout << "----------------------------------------------------" << endl;
	c = generateC(colGroups);
	c.t().raw_print();
	cout << "----------------------------------------------------" << endl << endl;

    cout << "Starting gurobi part" << endl;
    cout << "----------------------------------------------------" << endl;
	auto vars = gurobiPart(A, c, b);
    cout << "----------------------------------------------------" << endl << endl;

    cout << "We have now following code" << endl;
    cout << "----------------------------------------------------" << endl;
    int sum = 0;
	for (uint32_t i = 0; i < vars.size(); ++i)
	{
		sum += vars.at(i) * c.at(i);
	}

	const int d = sum - b;
    cout << "[n,k,d]_q" << endl;
    cout << "[" << sum << "," << k << "," << d << "]_" << q << "" << endl;
    cout << "----------------------------------------------------" << endl << endl;

    cout << "Generate Generatormatrix G" << endl;
    cout << "----------------------------------------------------" << endl;
	auto G = generateG(vars, colGroups, rVector);
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

int getMinHammingDistance(int q, int k, const s32_mat &G) {
    vector<s32_vec> language = generateLanguage(q, k);
    language.erase(language.begin());

    vector<s32_vec> codes;
    for (uint32_t i = 0; i < language.size() - 1; i++) {
	    const auto current = language.at(i);
	    const auto code = encode(current, G, q);
        codes.push_back(code);
    }

	auto minDistance = numeric_limits<int>::max();
    for (uint32_t i = 0; i < codes.size() - 1; i++) {
        for (uint32_t j = i + 1; j < codes.size(); j++) {
	        const auto left = codes.at(i);
	        const auto right = codes.at(j);
	        const auto distance = getHammingDistance(left, right);
            if (distance > 0 && distance < minDistance) {
                minDistance = distance;
            }
        }
    }

    return minDistance;
}

s32_vec encode(const s32_vec &row, const s32_mat &G, int q) {
    arma::s32_rowvec code = row.t() * G;
    for (arma::uword i = 0; i < code.size(); i++) {
        code.at(i) = static_cast<int>(code.at(i)) % q;
    }
    return code.t();
}

int getHammingDistance(const s32_vec &left, const s32_vec &right) {
    if (left.size() != right.size())
        return -1;

    int diff = 0;
    for (arma::uword i = 0; i < left.size(); i++) {
	    const int l = left.at(i);
	    const int r = right.at(i);
        if (l < r) {
            diff += r - l;
        } else {
            diff += l - r;
        }
    }

    return diff;
}

vector<s32_vec> generateLanguage(int q, int k) {
    vector<s32_vec> rVector = canonicalRepresent(q, k + 1);

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

s32_mat generateG(vector<double> xVec, vector<vector<arma::uword>> colGroups, const vector<s32_vec> &rVector)
{
    int colSize = 0;
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

s32_mat generateE0(int k)
{
	/*
	 Found Producer e0:
	1 0 0 0 0
	1 1 1 1 0
	1 0 1 0 0
	1 0 0 1 0
	0 0 1 0 1
	 */
	if (OPTIMIZE_GROUPS)
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
	else
	{
		return s32_mat(k, k);
	}
	/*s32_mat e0;
	e0 << 1 << 0 << 1 << arma::endr
	<< 0 << 1 << 0 << arma::endr
	<< 0 << 0 << 1 << arma::endr;

	return e0;*/

	/*s32_mat e0;
	e0 << 1 << 0 << 0 << arma::endr
		<< -2 << 0 << 1 << arma::endr
		<< 1 << -1 << 0 << arma::endr;

	return e0;*/
}

void normalizeVector(s32_vec& v, int q)
{
	for (auto& value : v)
	{
		value %= q;
	}
}

bool isEqualVector(s32_vec& v1, s32_vec& v2)
{
	bool ret = false;

	if (v1.n_rows == v2.n_rows)
	{
		ret = true;
		for (arma::uword i = 0; i < v1.n_rows; ++i)
		{
			if (v1.at(i) != v2.at(i))
			{
				ret = false;
			}
		}
	}

	return ret;
}

bool isLinDependent(s32_vec& v1, s32_vec v2, int q)
{
	bool ret = false;

	if (v1.n_cols == v2.n_cols)
	{
		for (auto i = 1; i < q; ++i)
		{
			v2 *= i;
			normalizeVector(v2, q);
			if (isEqualVector(v1, v2))
			{
				ret = true;
				break;
			}
		}
	}

	return ret;
}

bool isNullVector(s32_vec& rVector)
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

int getVectorIndex(vector<s32_vec>& rVector, s32_vec& rVec, int q)
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

vector<vector<arma::uword>> cycleCheck(vector<s32_vec>& rVector, s32_mat& E, int q)
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

	for (auto it = cycles.begin(); it != cycles.end(); )
	{
		if ((*it).size() > 2 && (*it).front() != (*it).back())
		{
			cycles.erase(it);
		}
		else
		{
			++it;
		}
	}

	return cycles;
}

vector<double> gurobiPart(const s32_mat &A, const s32_vec& c, int b) 
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

        vector<double> results;
        for (auto& modelVar : modelVars)
        {
            cout << modelVar.get(GRB_StringAttr_VarName) << " "
                 << modelVar.get(GRB_DoubleAttr_X) << endl;
            results.push_back(modelVar.get(GRB_DoubleAttr_X));
        }

        return results;
    } catch (const GRBException& e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
    } catch (...) {
        cout << "Exception during optimization" << endl;
    }
    return vector<double>();
}

s32_vec generateC(vector<vector<arma::uword>>& colGroups)
{
	s32_vec c(colGroups.size());

	for (arma::uword i = 0; i < colGroups.size(); ++i)
	{
		c.at(i) = colGroups.at(i).size();
	}

	return c;
}

s32_mat generateMatrixA(const vector<s32_vec> &rVector, int q) {
	s32_mat A(rVector.size(), rVector.size(), arma::fill::zeros);
    for (arma::uword i = 0; i < rVector.size(); i++) {
        for (arma::uword j = 0; j < rVector.size(); j++) {
            if (static_cast<int>(dot(rVector.at(i), rVector.at(j))) % q == 0) {
                A(i, j) = 1;
            }
        }
    }
    return A;
}

void regenerateMatrixA(s32_mat& A, vector<vector<arma::uword>>& colCycles, vector<vector<arma::uword>>& rowCycles)
{
	vector<arma::uword> colsDelete;
	vector<arma::uword> rowsDelete;

	for (auto& c : colCycles)
	{
		for (arma::uword i = 1; i < c.size() - 1; ++i)
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
		for (arma::uword i = 1; i < r.size() - 1; ++i)
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

vector<s32_vec> canonicalRepresent(int q, int k) {
    vector<s32_vec> rVector;
    s32_vec tmp = s32_vec(k);
    tmp.fill(0);
    rVector.push_back(tmp);
    rVector.at(0).at(k - 1) = 1;

    for (int i = k - 2; i >= 0; i--) {
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

bool sortHelper(vector<arma::uword> i, vector<arma::uword> j)
{
	return i.at(0) < j.at(0);
}

vector<vector<arma::uword>> generateRepresentGroups(vector<vector<arma::uword>>& colCycles, int vecCount)
{
	vector<vector<arma::uword>> repGroups;
	repGroups.reserve(colCycles.size());
	for (auto& colCycle : colCycles)
	{
		repGroups.push_back(colCycle);
	}

	for (auto& c : repGroups)
	{
		c.pop_back();
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

int recursionHelper(vector<s32_vec> &rVector, int vPosition, int hPosition, int q, int k) {
    s32_vec tmp = rVector.at(vPosition);
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
