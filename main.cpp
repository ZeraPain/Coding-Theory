#include <iostream>
#include <armadillo>
#include <gurobi_c++.h>

using std::vector;
using arma::s32_rowvec;
using std::cout;
using std::endl;
using arma::s32_mat;
using std::string;
using std::to_string;
using std::numeric_limits;

const int q = 3;

const int k = 3;

const int b = 3;

vector<s32_rowvec> canonicalRepresent(int q, int k);

int recursionHelper(vector<s32_rowvec> &rVector, int vPosition, int hPosition, int q, int k);

s32_mat generateMatrixA(const vector<s32_rowvec> &rVector, int q);

void regenerateMatrixA(s32_mat& A, std::vector<std::vector<arma::uword>>& cycles);

vector<double> gurobiPart(const s32_mat &A, const s32_rowvec& c, int b);

s32_rowvec generateC(s32_mat& A);

s32_mat generateG(vector<double> xMax, vector<s32_rowvec> rVector);

s32_mat generateE0();

void normalizeVector(s32_rowvec& rVector, int q);

bool isEqualVector(s32_rowvec& rVector1, s32_rowvec& rVector2);

int getVectorIndex(vector<s32_rowvec>& rVector, s32_rowvec& rVec);

std::vector<std::vector<arma::uword>> cycleCheck(vector<s32_rowvec>& rVector, s32_mat& E, int q);

int getMinHammingDistance(int q, int k, const s32_mat &G);

vector<s32_rowvec> generateLanguage(int q, int k);

int getHammingDistance(const s32_rowvec &left, const s32_rowvec &right);

s32_rowvec encode(const s32_rowvec &row, const s32_mat &G, int q);

int main() {
    cout << "q = " << q << endl << "k = " << k << endl << "b = " << b << endl;
    cout << "####################################################" << endl << endl;

    cout << "Generate canonical represents" << endl;
    cout << "----------------------------------------------------" << endl;
	auto rVector = canonicalRepresent(q, k);
    for (arma::uword i = 0; i < rVector.size(); ++i)
    {
		cout << "r" << i << "\t";
		rVector.at(i).raw_print();
    }
    cout << "----------------------------------------------------" << endl << endl;

	cout << "Generate matrix A" << endl;
	cout << "----------------------------------------------------" << endl;
	auto A = generateMatrixA(rVector, q);
	A.raw_print();
	cout << "----------------------------------------------------" << endl << endl;

	cout << "Check for cylces" << endl;
	cout << "----------------------------------------------------" << endl;
	auto e0 = generateE0();
	auto cycles = cycleCheck(rVector, e0, q);
	for (auto& c : cycles)
	{
		cout << "Found cycle: ";
		for (arma::uword i = 0; i < c.size(); i++)
		{
			cout << "r" << c.at(i);
			if (i != c.size() - 1) cout << " -> ";
		}
		cout << endl;
	}
	cout << "----------------------------------------------------" << endl << endl;

	s32_rowvec c(A.n_cols);
	c.fill(1);

	if (!cycles.empty())
	{
		cout << "Re-calculate matrix A with cylces" << endl;
		cout << "----------------------------------------------------" << endl;
		regenerateMatrixA(A, cycles);
		A.raw_print();
		cout << "----------------------------------------------------" << endl << endl;

		cout << "Calculate vector c" << endl;
		cout << "----------------------------------------------------" << endl;
		c = generateC(A);
		c.raw_print();
		cout << "----------------------------------------------------" << endl << endl;
	}

    cout << "Starting gurobi part" << endl;
    cout << "----------------------------------------------------" << endl;
	auto vars = gurobiPart(A, c, b);
    cout << "----------------------------------------------------" << endl << endl;

    cout << "We have now following code" << endl;
    cout << "----------------------------------------------------" << endl;
    int sum = 0;
    for (auto var : vars)
    {
        sum += var;
    }
	const int d = sum - b;
    cout << "[n,k,d]_q" << endl;
    cout << "[" << sum << "," << k << "," << d << "]_" << q << "" << endl;
    cout << "----------------------------------------------------" << endl << endl;

    cout << "Generate Generatormatrix G" << endl;
    cout << "----------------------------------------------------" << endl;
	auto G = generateG(vars, rVector);
    G.raw_print();
    cout << "----------------------------------------------------" << endl << endl;

    cout << "Proof Hamming Distance" << endl;
    cout << "----------------------------------------------------" << endl;
	const int minDistance = getMinHammingDistance(q, k, G);
    cout << minDistance << " = calculated distance" << endl;
    cout << d << " = should be the distance" << endl;
    cout << "----------------------------------------------------" << endl << endl;

	getchar();

    return 0;
}

int getMinHammingDistance(int q, int k, const s32_mat &G) {
    vector<s32_rowvec> language = generateLanguage(q, k);

    language.erase(language.begin());


    vector<s32_rowvec> codes;
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
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
    }

    return minDistance;
}

s32_rowvec encode(const s32_rowvec &row, const s32_mat &G, int q) {
    s32_rowvec code = row * G;
    for (arma::uword i = 0; i < code.size(); i++) {
        code.at(i) = static_cast<int>(code.at(i)) % q;
    }
    return code;
}

int getHammingDistance(const s32_rowvec &left, const s32_rowvec &right) {
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

vector<s32_rowvec> generateLanguage(int q, int k) {
    vector<s32_rowvec> rVector = canonicalRepresent(q, k + 1);

    vector<s32_rowvec> language;
    for (auto current : rVector)
    {
	    s32_rowvec newtmp(k);
        if (current.at(0) == 1) {
            for (int j = 1; j < k + 1; j++) {
                newtmp[j - 1] = current[j];
            }
            language.push_back(newtmp);
        }
    }

    return language;
}

s32_mat generateG(vector<double> xMax, vector<s32_rowvec> rVector) {
    int colSize = 0;
    for (auto i : xMax)
    {
        colSize += i;
    }
	const auto rowSize = rVector.at(0).size();
	s32_mat G(rowSize, colSize, arma::fill::zeros);

    int counter = 0;
    for (uint32_t i = 0; i < xMax.size(); i++) {
        for (uint32_t j = 0; j < xMax.at(i); j++) {
            for (arma::uword p = 0; p < rowSize; p++) {
                G(p, counter) = rVector.at(i).at(p);
            }
            counter++;
        }
    }

    return G;
}

s32_mat generateE0()
{
	s32_mat e0;
	e0 << 1 << 0 << 1 << arma::endr
		<< 0 << 1 << 0 << arma::endr
		<< 0 << 0 << 1 << arma::endr;

	return e0;
}

void normalizeVector(s32_rowvec& rVector, int q)
{
	for (auto& value : rVector)
	{
		value %= q;
	}
}

bool isEqualVector(s32_rowvec& rVector1, s32_rowvec& rVector2)
{
	auto ret = true;

	if (rVector1.size() == rVector2.size())
	{
		for (arma::uword i = 0; i < rVector1.size(); i++)
		{
			if (rVector1.at(i) != rVector2.at(i))
			{
				ret = false;
			}
		}
	}

	return ret;
}

int getVectorIndex(vector<s32_rowvec>& rVector, s32_rowvec& rVec)
{
	auto ret = -1;

	for (arma::uword i = 0; i < rVector.size(); i++)
	{
		if (isEqualVector(rVector.at(i), rVec))
		{
			ret = i;
			break;
		}
	}
	return ret;
}

std::vector<std::vector<arma::uword>> cycleCheck(vector<s32_rowvec>& rVector, s32_mat& E, int q)
{
	std::vector<std::vector<arma::uword>> cycles;

	for (arma::uword i = 0; i < rVector.size(); i++)
	{
		s32_rowvec tmpVec = rVector.at(i) * E;
		normalizeVector(tmpVec, q);
		if (!isEqualVector(rVector.at(i), tmpVec))
		{
			const auto vectorIndex = getVectorIndex(rVector, tmpVec);
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
				}

				if (!isInQueue)
				{
					std::vector<arma::uword> tmpQueue;
					tmpQueue.push_back(i);
					tmpQueue.push_back(vectorIndex);
					cycles.push_back(tmpQueue);
				}
			}
		}
	}

	for (auto it = cycles.begin(); it != cycles.end(); ++it)
	{
		if ((*it).front() != (*it).back())
		{
			cycles.erase(it);
		}
	}

	return cycles;
}

vector<double> gurobiPart(const s32_mat &A, const s32_rowvec& c, int b) 
{
    try {
	    const auto env = GRBEnv();
	    auto model = GRBModel(env);
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

		model.update();
		model.write("debug.lp");
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

s32_rowvec generateC(s32_mat& A)
{
	s32_rowvec c(A.n_cols);

	for (arma::uword i = 0; i < A.n_cols; i++)
	{
		auto max = A(0, i);
		for (arma::uword j = 1; j < A.n_rows; j++)
		{
			if (A(j, i) > max)
			{
				max = A(j, i);
			}
		}
		c.at(i) = max;
	}

	return c;
}

s32_mat generateMatrixA(const vector<s32_rowvec> &rVector, int q) {
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

void regenerateMatrixA(s32_mat& A, std::vector<std::vector<arma::uword>>& cycles)
{
	for (auto it = cycles.rbegin(); it != cycles.rend(); ++it)
	{
		for (arma::uword i = (*it).size() - 2; i > 0; --i)
		{
			for (arma::uword j = 0; j < A.n_rows; j++)
			{
				A(j, (*it).at(0)) += A(j, (*it).at(i));
			}
			A.shed_col((*it).at(i));
		}

		for (arma::uword i = 0; i < (*it).size() -2; ++i)
		{
			A.shed_row((*it).at(i));
		}
	}
}

vector<s32_rowvec> canonicalRepresent(int q, int k) {
    vector<s32_rowvec> rVector;
    s32_rowvec tmp = s32_rowvec(k);
    tmp.fill(0);
    rVector.push_back(tmp);
    rVector.at(0).at(k - 1) = 1;

    for (int i = k - 2; i >= 0; i--) {
		s32_rowvec tmp2(k);
        for (int j = 0; j < k; j++) {
			tmp2.at(j) = 0;
        }
		tmp2.at(i) = 1;
        rVector.push_back(tmp2);

        recursionHelper(rVector, rVector.size() - 1, i + 1, q, k);
    }

    return rVector;
}

int recursionHelper(vector<s32_rowvec> &rVector, int vPosition, int hPosition, int q, int k) {
    s32_rowvec tmp = rVector.at(vPosition);
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