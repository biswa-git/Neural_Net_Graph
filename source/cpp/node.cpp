#include<node.hpp>

node::node() :id(count++), bias(0.0), activation_pointer(nullptr)
{
}

node::~node()
{
}

int node::count = 0;

int node::get_id()
{
	return id;
}

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

const double& node::get_bias() const
{
	return bias;
}
/*
void node::set_value(const Eigen::VectorXd& value)
{
	this->value = value;
}

const Eigen::VectorXd& node::get_value() const
{
	return value;
}
*/
void node::set_activation_value(const Eigen::VectorXd& value)
{
	this->activation_value = value;
}

const Eigen::VectorXd& node::get_activation_value() const
{
	return activation_value;
}

void node::set_derivative_value(const Eigen::VectorXd& value)
{
	this->derivative_value = value;
}

const Eigen::VectorXd& node::get_derivative_value() const {
	return derivative_value;
}

void node::set_activation(activation::activation* activation_pointer)
{
	this->activation_pointer = activation_pointer;
}

activation::activation* node::get_activation() const
{
	return activation_pointer;
}

void node::set_delta(const Eigen::VectorXd& delta) {
	this->delta = delta;
}

const Eigen::VectorXd& node::get_delta() const {
	return delta;
}

void node::set_chain(const Eigen::VectorXd& chain)
{
	this->chain = chain;
}

const Eigen::VectorXd& node::get_chain() const
{
	return chain;
}
