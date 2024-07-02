#include<node.hpp>

node::node() :id(count++), bias(0.0), activation_pointer(nullptr)
{
}

node::~node()
{
}

int node::count = 0;

std::vector<weight*>& node::get_back_weights()
{
	return back_weights;
}

std::vector<weight*>& node::get_front_weights()
{
	return front_weights;
}

void node::set_bias(const double& bias)
{
	this->bias = bias;
}

double& node::get_bias()
{
	return bias;
}

void node::set_value(const Eigen::VectorXd& value)
{
	this->value = value;
}

Eigen::VectorXd& node::get_value()
{
	return value;
}

void node::set_activation_value(const Eigen::VectorXd& value)
{
	this->activation_value = value;
}

Eigen::VectorXd& node::get_activation_value()
{
	return activation_value;
}

void node::set_derivative_value(const Eigen::VectorXd& value)
{
	this->derivative_value = value;
}

Eigen::VectorXd& node::get_derivatve_value()
{
	return derivative_value;
}

void node::set_activation(activation* activation_pointer)
{
	this->activation_pointer = activation_pointer;
}

activation* node::get_activation()
{
	return activation_pointer;
}
