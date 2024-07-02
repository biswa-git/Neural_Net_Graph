#pragma once
#include<Eigen/Core>


enum ACTIVATION
{
	LINEAR = 0,
	RELU,
	GELU
};

class activation
{
public:
	activation();
	~activation();
	virtual Eigen::VectorXd calculate(const Eigen::VectorXd&) = 0;
	virtual Eigen::VectorXd calculate_derivative(const Eigen::VectorXd& input) = 0;
private:

};

class linear_activation : public activation
{
public:
	linear_activation();
	~linear_activation();
	Eigen::VectorXd calculate(const Eigen::VectorXd&);
	Eigen::VectorXd calculate_derivative(const Eigen::VectorXd& input);

private:

};



class ReLU_activation : public activation
{
public:
	ReLU_activation();
	~ReLU_activation();
	Eigen::VectorXd calculate(const Eigen::VectorXd&);
	Eigen::VectorXd calculate_derivative(const Eigen::VectorXd& input);

private:

};
