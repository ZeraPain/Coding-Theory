#include <iostream>
#include <armadillo>
#include <numeric>

#include "LinearCodeGenerator.h"
#include "Math.h"
#include <gurobi_c++.h>

template<class T, class I1, class I2>
T hamming_distance(uint32_t size, I1 b1, I2 b2)
{
	return std::inner_product(b1, b1 + size, b2, T{},
		std::plus<T>(), std::not_equal_to<T>());
}

int getMinHammingDistance(const int q, const int k, const arma::s32_mat &G)
{
	auto language = Math::generateLanguage(q, k);
	language.erase(language.begin());

	std::vector<arma::s32_vec> codes;
	codes.reserve(language.size());
	for (const auto& word : language)
	{
		codes.push_back(Math::encode(word, G, q));
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

int main()
{
	/* Parameters */
	constexpr int q = 7;
	constexpr int k = 3;
	constexpr int b = 3;

	std::cout << "###### Linear Code Creator ######" << std::endl << std::endl;
	std::cout << "q = " << q << std::endl << "k = " << k << std::endl << "b = " << b << std::endl;

	const LinearCodeGenerator linCodeGenerator(q, k, b);

	try
	{
		const auto linCode = linCodeGenerator.generateCode();

		std::cout << "We have now following code" << std::endl;
		std::cout << "----------------------------------------------------" << std::endl;
		std::cout << "[n,k,d]_q" << std::endl;
		std::cout << "[" << linCode.n << "," << linCode.k << "," << linCode.d << "]_" << linCode.q << "" << std::endl;
		std::cout << "----------------------------------------------------" << std::endl << std::endl;
		std::cout << "Generatormatrix G" << std::endl;
		std::cout << "----------------------------------------------------" << std::endl;
		linCode.G.raw_print();
		std::cout << "----------------------------------------------------" << std::endl << std::endl;


		std::cout << "Proof Hamming Distance" << std::endl;
		std::cout << "----------------------------------------------------" << std::endl;
		const int minDistance = getMinHammingDistance(q, k, linCode.G);
		std::cout << minDistance << " = calculated distance" << std::endl;
		std::cout << linCode.d << " = should be the distance" << std::endl;
		std::cout << "----------------------------------------------------" << std::endl << std::endl;
	}
	catch (const GRBException& e) {
		std::cout << "Error code = " << e.getErrorCode() << std::endl;
		std::cout << e.getMessage() << std::endl;
	}
	catch (...) {
		std::cout << "Exception during optimization" << std::endl;
	}

	std::cout << "Press any key to exit...";
	std::cin.get();

	return 0;
}
