#include "Math.h"

namespace Math
{
	auto normalizeVector(arma::s32_vec& v, int q) -> void
	{
		for (auto& value : v)
		{
			value %= q;
		}
	}

	auto isEqualVector(const arma::s32_vec& v1, const arma::s32_vec& v2) -> bool
	{
		auto ret = false;

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

	auto isLinDependent(const arma::s32_vec& v1, const arma::s32_vec& v2, const uint16_t q) -> bool
	{
		auto ret = false;

		if (v1.size() == v2.size())
		{
			for (auto i = 1; i < q; ++i)
			{
				arma::s32_vec tmp = v2 * i;
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

	auto isNullVector(const arma::s32_vec& rVector) -> bool
	{
		auto ret = true;

		for (auto& r : rVector)
		{
			if (0 != r)
			{
				ret = false;
			}
		}

		return ret;
	}

	auto getVectorIndex(const std::vector<arma::s32_vec>& rVector, const arma::s32_vec& rVec, int q) -> int
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

	auto encode(const arma::s32_vec& word, const arma::s32_mat& G, int q) -> arma::s32_vec
	{
		arma::s32_vec encoded = (word.t() * G).t();
		normalizeVector(encoded, q);

		return encoded;
	}

	auto recursionHelper(std::vector<arma::s32_vec>& rVector, uint16_t vPosition, uint16_t hPosition, uint16_t q, uint16_t k) -> int
	{
		auto tmp = rVector.at(vPosition);
		rVector.pop_back();
		for (auto i = 0; i < q; i++)
		{
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

	auto generateCanonialRepresents(uint16_t q, uint16_t k) -> std::vector<arma::s32_vec>
	{
		std::vector<arma::s32_vec> crVector;

		auto tmp = arma::s32_vec(k);
		tmp.fill(0);
		crVector.push_back(tmp);
		crVector.at(0).at(k - 1) = 1;

		for (auto i = k - 2; i >= 0; i--)
		{
			arma::s32_vec tmp2(k);
			for (int j = 0; j < k; j++) {
				tmp2.at(j) = 0;
			}
			tmp2.at(i) = 1;
			crVector.push_back(tmp2);

			recursionHelper(crVector, crVector.size() - 1, i + 1, q, k);
		}

		return crVector;
	}

	auto generateLanguage(int q, int k) -> std::vector<arma::s32_vec>
	{
		auto rVector = generateCanonialRepresents(q, k + 1);

		std::vector<arma::s32_vec> language;
		for (auto current : rVector)
		{
			arma::s32_vec newtmp(k);
			if (current.at(0) == 1) {
				for (auto j = 1; j < k + 1; j++) {
					newtmp[j - 1] = current[j];
				}
				language.push_back(newtmp);
			}
		}

		return language;
	}
}
