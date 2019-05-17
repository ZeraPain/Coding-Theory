#pragma once
#include <armadillo>

namespace Math
{
	auto normalizeVector(arma::s32_vec& v, int q) -> void;
	auto isEqualVector(const arma::s32_vec& v1, const arma::s32_vec& v2) -> bool;
	auto isLinDependent(const arma::s32_vec& v1, const arma::s32_vec& v2, uint16_t q) -> bool;
	auto isNullVector(const arma::s32_vec& rVector) -> bool;
	auto getVectorIndex(const std::vector<arma::s32_vec>& rVector, const arma::s32_vec& rVec, int q) -> int;

	auto encode(const arma::s32_vec& word, const arma::s32_mat &G, int q) -> arma::s32_vec;
	auto generateCanonialRepresents(uint16_t q, uint16_t k)->std::vector<arma::s32_vec>;
	auto generateLanguage(int q, int k) -> std::vector<arma::s32_vec>;
}
