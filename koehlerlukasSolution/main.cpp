#include <iostream>
#include <armadillo>
#include <gurobi_c++.h>

using std::vector;
using arma::rowvec;
using std::cout;
using std::endl;
using arma::mat;
using std::string;
using std::to_string;
using std::numeric_limits;

const int q = 3;

const int k = 2;

const int b = 2;

vector<rowvec> canonicalRepresent(int q, int k);

int recursionHelper(vector<rowvec> &rVector, int vPosition, int hPosition, int q, int k);

mat generateMatrixA(const vector<rowvec> &rVector, int q);

vector<double> gurobiPart(const mat &A, int b);

mat generateG(vector<double> xMax, vector<rowvec> rVector);

int getMinHammingDistance(int q, int k, const mat &G);

vector<rowvec> generateLanguage(int q, int k);

int getHammingDistance(const rowvec &left, const rowvec &right);

rowvec encode(const rowvec &row, const mat &G, int q);

int main() {
    cout << "q = " << q << endl << "k = " << k << endl << "b = " << b << endl;
    cout << "####################################################" << endl << endl;

    cout << "Generate canonical represents" << endl;
    cout << "----------------------------------------------------" << endl;
    vector<rowvec> rVector = canonicalRepresent(q, k);
    for (int i = 0; i < rVector.size(); i++) {
        rVector.at(i).print();
    }
    cout << "----------------------------------------------------" << endl << endl;

    cout << "Generate matrix A" << endl;
    cout << "----------------------------------------------------" << endl;
    mat A = generateMatrixA(rVector, q);
    A.print();
    cout << "----------------------------------------------------" << endl << endl;

    cout << "Starting gurobi part" << endl;
    cout << "----------------------------------------------------" << endl;
    vector<double> vars = gurobiPart(A, b);
    cout << "----------------------------------------------------" << endl << endl;

    cout << "We have now following code" << endl;
    cout << "----------------------------------------------------" << endl;
    int sum = 0;
    for (int i = 0; i < vars.size(); i++) {
        sum += vars.at(i);
    }
    int d = sum - b;
    cout << "[n,k,d]_q" << endl;
    cout << "[" << sum << "," << k << "," << d << "]_" << q << "" << endl;
    cout << "----------------------------------------------------" << endl << endl;

    cout << "Generate Generatormatrix G" << endl;
    cout << "----------------------------------------------------" << endl;
    mat G = generateG(vars, rVector);
    G.print();
    cout << "----------------------------------------------------" << endl << endl;

    cout << "Proof Hamming Distance" << endl;
    cout << "----------------------------------------------------" << endl;
    int minDistance = getMinHammingDistance(q, k, G);
    cout << minDistance << " = calculated distance" << endl;
    cout << d << " = should be the distance" << endl;
    cout << "----------------------------------------------------" << endl << endl;

    return 0;
}

int getMinHammingDistance(int q, int k, const mat &G) {
    vector<rowvec> language = generateLanguage(q, k);

    language.erase(language.begin());


    vector<rowvec> codes;
    for (int i = 0; i < language.size() - 1; i++) {
        rowvec current = language.at(i);
        rowvec code = encode(current, G, q);
        codes.push_back(code);
    }

    int minDistance = numeric_limits<int>::max();
    for (int i = 0; i < codes.size() - 1; i++) {
        for (int j = i + 1; j < codes.size(); j++) {
            rowvec left = codes.at(i);
            rowvec right = codes.at(j);
            int distance = getHammingDistance(left, right);
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
    }

    return minDistance;
}

rowvec encode(const rowvec &row, const mat &G, int q) {
    rowvec code = row * G;
    for (int i = 0; i < code.size(); i++) {
        code.at(i) = (int) code.at(i) % q;
    }
    return code;
}

int getHammingDistance(const rowvec &left, const rowvec &right) {
    if (left.size() != right.size())
        return -1;

    int diff = 0;
    for (int i = 0; i < left.size(); i++) {
        int l = left.at(i);
        int r = right.at(i);
        if (l < r) {
            diff += r - l;
        } else {
            diff += l - r;
        }
    }

    return diff;
}

vector<rowvec> generateLanguage(int q, int k) {
    vector<rowvec> rVector = canonicalRepresent(q, k + 1);

    vector<rowvec> language;
    for (int i = 0; i < rVector.size(); i++) {
        rowvec current = rVector.at(i);
        rowvec newtmp(k);
        if (current.at(0) == 1) {
            for (int j = 1; j < k + 1; j++) {
                newtmp[j - 1] = current[j];
            }
            language.push_back(newtmp);
        }
    }

    return language;
}

mat generateG(vector<double> xMax, vector<rowvec> rVector) {
    int colSize = 0;
    for (int i = 0; i < xMax.size(); i++) {
        colSize += xMax.at(i);
    }
    int rowSize = rVector.at(0).size();
    mat G(rowSize, colSize, arma::fill::zeros);

    int counter = 0;
    for (int i = 0; i < xMax.size(); i++) {
        for (int j = 0; j < xMax.at(i); j++) {
            for (int p = 0; p < rowSize; p++) {
                G(p, counter) = rVector.at(i).at(p);
            }
            counter++;
        }
    }

    return G;
}

vector<double> gurobiPart(const mat &A, int b) {
    try {
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        vector<GRBVar> modelVars;

        for (int i = 0; i < A.n_cols; i++) {
            modelVars.push_back(model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER, string("var").append(to_string(i))));
        }

        GRBLinExpr expr = 0.0;

        for (int i = 0; i < A.n_cols; i++) {
            expr += modelVars.at(i);
        }

        model.setObjective(expr, GRB_MAXIMIZE);

        for (int x = 0; x < A.n_rows; x++) {
            GRBLinExpr expr = 0.0;
            for (int y = 0; y < A.n_cols; y++) {
                expr += modelVars.at(y) * A(x, y);
            }
            model.addConstr(expr, GRB_LESS_EQUAL, b, string("constr").append(to_string(x)));
        }

        model.optimize();

        vector<double> results;
        for (int i = 0; i < modelVars.size(); i++) {
            cout << modelVars.at(i).get(GRB_StringAttr_VarName) << " "
                 << modelVars.at(i).get(GRB_DoubleAttr_X) << endl;
            results.push_back(modelVars.at(i).get(GRB_DoubleAttr_X));
        }
        return results;

    } catch (GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
    } catch (...) {
        cout << "Exception during optimization" << endl;
    }
    return vector<double>();
}

mat generateMatrixA(const vector<rowvec> &rVector, int q) {
    mat A(rVector.size(), rVector.size(), arma::fill::zeros);
    for (int i = 0; i < rVector.size(); i++) {
        for (int j = 0; j < rVector.size(); j++) {
            if (static_cast<int>(dot(rVector.at(i), rVector.at(j))) % q == 0) {
                A(i, j) = 1;
            }
        }
    }
    return A;
}

vector<rowvec> canonicalRepresent(int q, int k) {
    vector<rowvec> rVector;
    rowvec tmp = rowvec(k);
    tmp.fill(0);
    rVector.push_back(tmp);
    rVector.at(0).at(k - 1) = 1;

    for (int i = k - 2; i >= 0; i--) {
        rowvec tmp(k);
        for (int j = 0; j < k; j++) {
            tmp.at(j) = 0;
        }
        tmp.at(i) = 1;
        rVector.push_back(tmp);

        recursionHelper(rVector, rVector.size() - 1, i + 1, q, k);
    }

    return rVector;
}

int recursionHelper(vector<rowvec> &rVector, int vPosition, int hPosition, int q, int k) {
    rowvec tmp = rVector.at(vPosition);
    rVector.pop_back();
    for (int i = 0; i < q; i++) {
        tmp.at(hPosition) = i;
        rVector.push_back(tmp);
        vPosition++;
        if (hPosition < k - 1) {
            int newVPosition = recursionHelper(rVector, vPosition - 1, hPosition + 1, q, k);
            vPosition = newVPosition;
        }
    }
    return vPosition;
}