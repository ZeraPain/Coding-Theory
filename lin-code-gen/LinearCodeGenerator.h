#pragma once
#include <armadillo>
#include <unordered_map>

struct LinearCode
{
	uint16_t n;
	uint16_t q;
	uint16_t k;
	uint16_t d;
	arma::s32_mat G;
};

class LinearCodeGenerator
{
public:
	LinearCodeGenerator(uint16_t q, uint16_t k, uint16_t b);
	auto generateCode() const -> LinearCode;

private:
	
	auto generateMatrixA(const std::vector<arma::s32_vec>& canonialRepresents) const -> arma::s32_mat;
	auto generateProducerE0(const std::vector<arma::s32_vec>& canonialRepresents, const arma::s32_mat& matrixA) const ->arma::s32_mat;
	auto generateOptimizedMatrixA(const arma::s32_mat& matrixA, const std::unordered_map<arma::uword, std::vector<arma::uword>>& colCycles, const std::unordered_map<arma::uword, std::vector<arma::uword>>& rowCycles) const->arma::s32_mat;
	static auto generateMatrixG(const arma::s32_vec& xVec, const std::vector<std::vector<arma::uword>>& colGroups, const std::vector<arma::s32_vec> &rVector) -> arma::s32_mat;

	static auto generateVectorC(const std::vector<std::vector<arma::uword>>& colGroups) -> arma::s32_vec;
	static auto generateVectorX(const arma::s32_mat &A, const arma::s32_vec& c, uint16_t b) -> arma::s32_vec;

	auto cycleCheck(const std::vector<arma::s32_vec>& rVector, const arma::s32_mat& E, uint16_t q) const -> std::unordered_map<arma::uword, std::vector<arma::uword>>;
	auto generateRepresentGroups(const std::unordered_map<arma::uword, std::vector<arma::uword>>& cycle, int vecCount) const -> std::vector<std::vector<arma::uword>>;
	auto generateMatrixE0() const -> arma::s32_mat;

	uint16_t m_q;
	uint16_t m_k;
	uint16_t m_b;

};

