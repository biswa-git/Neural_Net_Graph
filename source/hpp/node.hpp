#pragma once
#include<vector>
#include<Eigen/Core>
#include<activation.hpp>
class weight;

class node
{
public:
	node();
	~node();
	static int count;
	std::vector<weight*>& get_back_weights();
	std::vector<weight*>& get_front_weights();
	void set_bias(const double&);
	double& get_bias();
	void set_value(const Eigen::VectorXd&);
	Eigen::VectorXd& get_value();
	void set_activation_value(const Eigen::VectorXd&);
	Eigen::VectorXd& get_activation_value();
	void set_derivative_value(const Eigen::VectorXd&);
	Eigen::VectorXd& get_derivatve_value();
	void set_activation(activation*);
	activation* get_activation();
private:
	int id;
	std::vector<weight*> back_weights;
	std::vector<weight*> front_weights;
	double bias;
	Eigen::VectorXd value;
	Eigen::VectorXd activation_value;
	Eigen::VectorXd derivative_value;
	activation* activation_pointer;
};