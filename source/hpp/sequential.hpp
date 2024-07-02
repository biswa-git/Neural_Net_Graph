#pragma once
#include<layer.hpp>

class sequential
{
public:
	sequential(const std::vector<std::vector<int>>&);
	~sequential();

	static linear_activation linear;
	static ReLU_activation ReLU;

	void fit(const std::vector< Eigen::VectorXd>&, const std::vector< Eigen::VectorXd>&, const int& = 1);
private:
	std::vector<layer*> layers;
	std::vector< Eigen::VectorXd> x;
	std::vector< Eigen::VectorXd> y;
	int batch_size;


};