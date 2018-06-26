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
using std::cerr;
using arma::cross;
using arma::vec;
using std::find_if;
using arma::mtGlue;
using std::exception;

const int q = 17;

const int k = 3;

const int b = 11;

const int Amin = 14;

const int Amax = 50;

const int AfindingTimeout = 30;

const int gurobiTimeout = 5;

const int nMin = 200;

const int programTimeout = 240;

const bool printSComponents = false;

const bool printAMatrix = false;

const bool printErzeuger = false;

const bool printGurobi = false;

vector<rowvec> canonicalRepresent(int q, int k);

int recursionHelper(vector<rowvec> &rVector, int vPosition, int hPosition, int q, int k);

mat generateMatrixA(const vector<rowvec> &rVector, int q);

vector<double> gurobiPart(const mat &A, int b, rowvec c, bool withC);

mat generateG(vector<double> xMax, vector<rowvec> rVector);

int getMinHammingDistance(int q, int k, const mat &G);

vector<rowvec> generateLanguage(int q, int k);

int getHammingDistance(const rowvec &left, const rowvec &right);

rowvec encode(const rowvec &row, const mat &G, int q);

vector<mat> getErzeuger(int k, int q);

vector<vector<rowvec>> getSComponents(const vector<rowvec> &rVector, const vector<mat> &eVec, bool eTransposed);

vector<rowvec>
getGraphConnections(rowvec row, const vector<mat> &eVec, const vector<rowvec> &rVector, vector<rowvec> &done,
                    bool transposed);

vector<rowvec> getLinearIndependent(mat row, const vector<rowvec> &rVector);

void module(mat &matrix);

void module(vec &myVec);

bool contains(const vector<rowvec> &myVector, rowvec row);

mat generateMatrixA2(const vector<vector<rowvec>> &s, const vector<vector<rowvec>> &sT);

rowvec generateVectorC(vector<vector<rowvec>> &vector);

mat generateGPart4(const vector<vector<rowvec>> &s, const vector<double> &X, int n);

template<class T, class I1, class I2>
T hamming_distance(size_t size, I1 b1, I2 b2);

int getNewHammingDistance(rowvec left, rowvec right);

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

    vector<vector<rowvec>> sTransposedComponents;
    vector<vector<rowvec>> sComponents;
    mat A2;
    vector<double> varsNew;
    mat G2;
    int minDistance;

    time_t start, ende, programStart, programEnd;
    time(&programStart);

    int d, sum = 0;
    int aTried = 0;
    int codeGenerated = 0;

    bool noCode = true;
    bool noAMatrix = false;

    int maxN = 0;

    do {
        time(&start);
        do {
            aTried++;
            cout << "Get Erzeuger" << endl;
            cout << "----------------------------------------------------" << endl;
            vector<mat> eVec = getErzeuger(k, q);
            if (printErzeuger) {
                for (mat current : eVec) {
                    current.print();
                    cout << endl;
                }
            }
            cout << "----------------------------------------------------" << endl << endl;

            cout << "Get S Components" << endl;
            cout << "----------------------------------------------------" << endl;
            sComponents = getSComponents(rVector, eVec, false);
            if (printSComponents) {
                for (vector<rowvec> currentVec : sComponents) {
                    for (rowvec current : currentVec) {
                        current.print();
                    }
                    cout << endl;
                }
            }
            cout << "----------------------------------------------------" << endl << endl;

            cout << "Get S Transposed Components" << endl;
            cout << "----------------------------------------------------" << endl;
            sTransposedComponents = getSComponents(rVector, eVec, true);
            if (printSComponents) {
                for (vector<rowvec> currentVec : sTransposedComponents) {
                    for (rowvec current : currentVec) {
                        current.print();
                    }
                    cout << endl;
                }
            }
            cout << "----------------------------------------------------" << endl << endl;

            cout << "Generate matrix A" << endl;
            cout << "----------------------------------------------------" << endl;
            A2 = generateMatrixA2(sComponents, sTransposedComponents);
            if (printAMatrix) {
                A2.print();
            }
            cout << "----------------------------------------------------" << endl << endl;

            time(&ende);
            ende -= start;

            time(&programEnd);
            programEnd -= programStart;

            if (programEnd >= programTimeout) {
                cout << "Exit searching after " << programTimeout << " seconds" << endl;
                break;
            } else if (ende >= AfindingTimeout) {
                cout << "No A matrix found after " << AfindingTimeout << " seconds" << endl;
                noAMatrix = true;
                break;
            } else if (A2.n_cols >= Amin && A2.n_cols <= Amax) {
                break;
            } else {
                cout << "Matrix A has wrong size. Trying again" << endl;
            }
        } while (true);

        if (noAMatrix) {
            break;
        }

        codeGenerated++;

        cout << "Generate vector c" << endl;
        cout << "----------------------------------------------------" << endl;
        rowvec c = generateVectorC(sComponents);
        c.print();
        cout << "----------------------------------------------------" << endl << endl;

        cout << "Starting gurobi part" << endl;
        cout << "----------------------------------------------------" << endl;
        varsNew = gurobiPart(A2, b, c, true);
        cout << "----------------------------------------------------" << endl << endl;

        cout << "We have now following code" << endl;
        cout << "----------------------------------------------------" << endl;
        sum = 0;
        for (int i = 0; i < varsNew.size(); i++) {
            sum += varsNew.at(i) * c.at(i);
        }
        d = sum - b;
        cout << "[n,k,d]_q" << endl;
        cout << "[" << sum << "," << k << "," << d << "]_" << q << "" << endl;
        cout << "----------------------------------------------------" << endl << endl;

        if (sum > maxN) {
            noCode = false;
            maxN = sum;
            G2 = generateGPart4(sComponents, varsNew, sum);
        }

        time(&programEnd);
        programEnd -= programStart;

        if (programEnd >= programTimeout) {
            cout << "Exit searching after " << programTimeout << " seconds" << endl;
            break;
        } else if (sum >= nMin) {
            cout << "Code found" << endl;
            break;
        }

    } while (true);

    if (maxN > 0 && noCode == false) {
        cout << "Generatormatrix G" << endl;
        cout << "----------------------------------------------------" << endl;
        G2.print();
        cout << "----------------------------------------------------" << endl << endl;

        cout << "Hamming Distance" << endl;
        cout << "----------------------------------------------------" << endl;
        d = maxN - b;
        minDistance = getMinHammingDistance(q, k, G2);
        cout << minDistance << " = calculated distance" << endl;
        cout << d << " = should be the distance" << endl;
        cout << "----------------------------------------------------" << endl << endl;
    }

    cout << "Summary" << endl;
    cout << "----------------------------------------------------" << endl;
    cout << "New A created" << endl;
    cout << aTried << " times" << endl << endl;
    cout << "Matrix A size" << endl;
    cout << A2.n_cols << endl << endl;
    if (maxN > 0) {
        cout << "We have now following code" << endl;
        d = maxN - b;
        cout << "[n,k,d]_q" << endl;
        cout << "[" << maxN << "," << k << "," << d << "]_" << q << "" << endl << endl;
    }
    cout << "Code generated" << endl;
    cout << codeGenerated << " times" << endl << endl;
    cout << "Max N" << endl;
    cout << maxN << endl << endl;
    time(&programEnd);
    programEnd -= programStart;
    cout << "It took" << endl;
    cout << programEnd << " seconds" << endl;
    cout << "----------------------------------------------------" << endl << endl;

    return 0;
}

mat generateGPart4(const vector<vector<rowvec>> &s, const vector<double> &X, int n) {
    mat G(s.at(0).at(0).size(), n, arma::fill::zeros);
    int counter = 0;
    for (int i = 0; i < X.size(); i++) {
        for (int j = 0; j < X.at(i); j++) {
            vector<rowvec> tmp = s.at(i);
            for (rowvec current : tmp) {
                for (int x = 0; x < current.size(); x++) {
                    G(x, counter) = current.at(x);
                }
                counter++;
            }
        }
    }

    return G;
}

template<class T, class I1, class I2>
T hamming_distance(size_t size, I1 b1, I2 b2) {
    return std::inner_product(b1, b1 + size, b2, T{},
                              std::plus<T>(), std::not_equal_to<T>());
}

rowvec generateVectorC(vector<vector<rowvec>> &s) {
    rowvec c(s.size(), arma::fill::zeros);
    for (int i = 0; i < s.size(); i++) {
        c.at(i) = s.at(i).size();
    }
    return c;
}

mat generateMatrixA2(const vector<vector<rowvec>> &s, const vector<vector<rowvec>> &sT) {
    mat A2(sT.size(), s.size(), arma::fill::zeros);
    for (int i = 0; i < sT.size(); i++) {
        for (int j = 0; j < s.size(); j++) {
            for (int x = 0; x < s.at(j).size(); x++) {
                if (static_cast<int>(dot(sT.at(i).at(0), s.at(j).at(x))) % q == 0) {
                    A2(i, j)++;
                }
            }
        }
    }
    return A2;
}

vector<vector<rowvec>> getSComponents(const vector<rowvec> &rVector, const vector<mat> &eVec, bool eTransposed) {
    vector<rowvec> done;
    vector<vector<rowvec>> returnValue;
    for (rowvec current : rVector) {
        if (contains(done, current)) {
            continue;
        }
        done.push_back(current);
        vector<rowvec> graphConnections = getGraphConnections(current, eVec, rVector, done, eTransposed);
        returnValue.push_back(graphConnections);
    }
    return returnValue;
}

bool contains(const vector<rowvec> &myVector, rowvec row) {
    return find_if(myVector.begin(), myVector.end(), [row](const rowvec &l) {
        int min = (l == row).min();
        if (min == 0) {
            return false;
        }
        return true;
    }) != myVector.end();
}

vector<rowvec>
getGraphConnections(rowvec row, const vector<mat> &eVec, const vector<rowvec> &rVector, vector<rowvec> &done,
                    bool transposed) {
    vector<rowvec> Q;
    vector<rowvec> L;
    Q.push_back(row);
    while (Q.size() > 0) {
        rowvec current = Q.back();
        Q.pop_back();
        L.push_back(current);
        for (mat e : eVec) {
            if (transposed) {
                e = e.t();
            }
            mat result = e * current.t();
            module(result);
            vector<rowvec> resultLinearIndependent = getLinearIndependent(result, rVector);
            if (resultLinearIndependent.size() > 1) {
                cerr << "more then 1 linear Independent" << endl;
            } else if (resultLinearIndependent.size() == 1) {
                if (!contains(L, resultLinearIndependent.at(0)) && !contains(done, resultLinearIndependent.at(0))) {
                    Q.push_back(resultLinearIndependent.at(0));
                    done.push_back(resultLinearIndependent.at(0));
                }
            }
        }
    }

    return L;
}

void module(mat &matrix) {
    for (int x = 0; x < matrix.n_cols; x++) {
        for (int y = 0; y < matrix.n_rows; y++) {
            matrix(y, x) = (int) matrix(y, x) % q;
        }
    }
}

void module(vec &myVec) {
    for (int i = 0; i < myVec.size(); i++) {
        myVec.at(i) = (int) myVec.at(i) % q;
    }
}

vector<rowvec> getLinearIndependent(mat row, const vector<rowvec> &rVector) {
    vector<rowvec> result;
    for (rowvec current : rVector) {
        vec a = row.col(0);
        vec b = current.t();
        for (int i = 1; i < q; i++) {
            vec tmp = i * a;
            module(tmp);
            int min = (tmp == b).min();
            if (min != 0) {
                result.push_back(current);
                break;
            }
        }
    }
    return result;
}

vector<mat> getErzeuger(int k, int q) {
    std::random_device generator;
    std::mt19937 mt(generator());
    std::uniform_int_distribution<int> distribution(0, q - 1);

    mat e(k, k);
    do {
        for (auto i = 0; i < k; ++i) {
            for (auto j = 0; j < k; ++j) {
                e(i, j) = distribution(generator);
            }
        }
    } while (0 == det(e));

    return {e};
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
            int distance = getNewHammingDistance(left, right);
            if (distance < minDistance) {
                minDistance = distance;
            }
        }
    }

    return minDistance;
}

int getNewHammingDistance(rowvec left, rowvec right) {
    return hamming_distance<int32_t>(left.size(), left.begin(), right.begin());
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

vector<double> gurobiPart(const mat &A, int b, rowvec c, bool withC) {
    try {
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        model.set(GRB_IntParam_CSIdleTimeout, gurobiTimeout);
        model.set(GRB_IntParam_ServerTimeout, gurobiTimeout);
        model.getEnv().set(GRB_IntParam_OutputFlag, printGurobi);

        vector<GRBVar> modelVars;
        modelVars.reserve(c.size());
        for (int i = 0; i < c.size(); i++) {
            modelVars.push_back(model.addVar(0.0, 1.0, 0.0, GRB_INTEGER));
        }

        GRBLinExpr expr = 0.0;
        for (int i = 0; i < c.size(); i++) {
            expr += modelVars.at(i) * c.at(i);
        }

        model.setObjective(expr, GRB_MAXIMIZE);

        for (int x = 0; x < A.n_rows; x++) {
            expr = 0.0;
            for (int y = 0; y < A.n_cols; y++) {
                expr += modelVars.at(y) * A(x, y);
            }
            model.addConstr(expr, GRB_LESS_EQUAL, b);
        }

        //model.update();
        //model.write("debug.lp");
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
    } catch (exception e) {
        cout << "Error code = " << e.what();
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