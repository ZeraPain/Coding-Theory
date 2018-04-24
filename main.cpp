#include <iostream>
#include <armadillo>

// Armadillo documentation is available at:
// http://arma.sourceforge.net/docs.html

void calc_lin_independ_vec(std::vector<arma::vec>& ret, arma::vec* init_vec, int q_index, const int q, int k_index, const int k)
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

std::vector<arma::vec> get_lin_independ_vector(const int q, const int k)
{
	std::vector<arma::vec> ret;

	for (auto k_index = k - 1; k_index >= 0; --k_index)
	{
		//std::cout << "stage " << k_index << std::endl;
		arma::vec init_vec(k); // set size to k
		init_vec.fill(0); // init with zero
		init_vec.at(k_index) = 1;
		calc_lin_independ_vec(ret, &init_vec, 1, q, k_index, k);
	}

	return ret;
}

int main()
{
	std::cout << "Armadillo version: " << arma::arma_version::as_string() << std::endl << std::endl;

	int q, k;

	std::cout << "Enter value for q: ";
	std::cin >> q;

	std::cout << "Enter value for k: ";
	std::cin >> k;

	auto lin_independ_vecs = get_lin_independ_vector(q, k);
	const int mat_size = (pow(q, k) - 1) / (q - 1);

	arma::mat A(mat_size, mat_size);
	for (auto i = 0; i < mat_size; ++i)
	{
		for (auto j = 0; j < mat_size; ++j)
		{
			const auto res = static_cast<int>(dot(lin_independ_vecs.at(i), lin_independ_vecs.at(j))) % q;
			A(i, j) = 0 == res ? 1 : 0;
		}
	}

	A.print("A:");

	system("PAUSE");
	return 0;

	/*cout << "Armadillo version: " << arma_version::as_string() << endl;

	mat A(2, 3);  // directly specify the matrix size (elements are uninitialised)

	cout << "A.n_rows: " << A.n_rows << endl;  // .n_rows and .n_cols are read only
	cout << "A.n_cols: " << A.n_cols << endl;

	A(1, 2) = 456.0;  // directly access an element (indexing starts at 0)
	A.print("A:");

	A = 5.0;         // scalars are treated as a 1x1 matrix
	A.print("A:");

	A.set_size(4, 5); // change the size (data is not preserved)

	A.fill(5.0);     // set all elements to a particular value
	A.print("A:");

	// endr indicates "end of row"
	A << 0.165300 << 0.454037 << 0.995795 << 0.124098 << 0.047084 << endr
		<< 0.688782 << 0.036549 << 0.552848 << 0.937664 << 0.866401 << endr
		<< 0.348740 << 0.479388 << 0.506228 << 0.145673 << 0.491547 << endr
		<< 0.148678 << 0.682258 << 0.571154 << 0.874724 << 0.444632 << endr
		<< 0.245726 << 0.595218 << 0.409327 << 0.367827 << 0.385736 << endr;

	A.print("A:");

	// determinant
	cout << "det(A): " << det(A) << endl;

	// inverse
	cout << "inv(A): " << endl << inv(A) << endl;

	// save matrix as a text file
	A.save("A.txt", raw_ascii);

	// load from file
	mat B;
	B.load("A.txt");

	// submatrices
	cout << "B( span(0,2), span(3,4) ):" << endl << B(span(0, 2), span(3, 4)) << endl;

	cout << "B( 0,3, size(3,2) ):" << endl << B(0, 3, size(3, 2)) << endl;

	cout << "B.row(0): " << endl << B.row(0) << endl;

	cout << "B.col(1): " << endl << B.col(1) << endl;

	// transpose
	cout << "B.t(): " << endl << B.t() << endl;

	// maximum from each column (traverse along rows)
	cout << "max(B): " << endl << max(B) << endl;

	// maximum from each row (traverse along columns)
	cout << "max(B,1): " << endl << max(B, 1) << endl;

	// maximum value in B
	cout << "max(max(B)) = " << max(max(B)) << endl;

	// sum of each column (traverse along rows)
	cout << "sum(B): " << endl << sum(B) << endl;

	// sum of each row (traverse along columns)
	cout << "sum(B,1) =" << endl << sum(B, 1) << endl;

	// sum of all elements
	cout << "accu(B): " << accu(B) << endl;

	// trace = sum along diagonal
	cout << "trace(B): " << trace(B) << endl;

	// generate the identity matrix
	mat C = eye<mat>(4, 4);

	// random matrix with values uniformly distributed in the [0,1] interval
	mat D = randu<mat>(4, 4);
	D.print("D:");

	// row vectors are treated like a matrix with one row
	rowvec r;
	r << 0.59119 << 0.77321 << 0.60275 << 0.35887 << 0.51683;
	r.print("r:");

	// column vectors are treated like a matrix with one column
	vec q;
	q << 0.14333 << 0.59478 << 0.14481 << 0.58558 << 0.60809;
	q.print("q:");

	// convert matrix to vector; data in matrices is stored column-by-column
	vec v = vectorise(A);
	v.print("v:");

	// dot or inner product
	cout << "as_scalar(r*q): " << as_scalar(r*q) << endl;

	// outer product
	cout << "q*r: " << endl << q * r << endl;

	// multiply-and-accumulate operation (no temporary matrices are created)
	cout << "accu(A % B) = " << accu(A % B) << endl;

	// example of a compound operation
	B += 2.0 * A.t();
	B.print("B:");

	// imat specifies an integer matrix
	imat AA;
	imat BB;

	AA << 1 << 2 << 3 << endr << 4 << 5 << 6 << endr << 7 << 8 << 9;
	BB << 3 << 2 << 1 << endr << 6 << 5 << 4 << endr << 9 << 8 << 7;

	// comparison of matrices (element-wise); output of a relational operator is a umat
	umat ZZ = (AA >= BB);
	ZZ.print("ZZ:");

	// cubes ("3D matrices")
	cube Q(B.n_rows, B.n_cols, 2);

	Q.slice(0) = B;
	Q.slice(1) = 2.0 * B;

	Q.print("Q:");

	// 2D field of matrices; 3D fields are also supported
	field<mat> F(4, 3);

	for (uword col = 0; col < F.n_cols; ++col)
		for (uword row = 0; row < F.n_rows; ++row)
		{
			F(row, col) = randu<mat>(2, 3);  // each element in field<mat> is a matrix
		}

	F.print("F:");*/
}

