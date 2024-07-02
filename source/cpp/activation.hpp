#pragma once
#include<Params.hpp>
class Activation
{
public:
	Activation();
	~Activation();
	static void Linear(Eigen::VectorXd&);
	static void ReLU(Eigen::VectorXd&);
	static void GeLU(Eigen::VectorXd&);
	static void Sigmoid(Eigen::VectorXd&);
private:

};
