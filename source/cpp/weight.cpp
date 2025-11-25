#include<weight.hpp>
#include <random>
#include <iostream>
weight::weight(node* back_node, node* front_node) 
	:back_node(back_node), front_node(front_node), value(1.0), delta(0.0), 
	 first_momentum(0.0), second_momentum(0.0)
{
	back_node->get_front_weights().emplace_back(this);
	front_node->get_back_weights().emplace_back(this);

	std::random_device rd;
	std::mt19937 gen(rd());

	// Xavier initialization: N(0, sqrt(2 / (fan_in + fan_out)))
	// Approximate with sqrt(2 / fan_in) for He initialization
	double fan_in = back_node ? 256.0 : 1.0;  // Rough estimate; ideally tracked per layer
	double std_dev = sqrt(2.0 / fan_in);
	
	std::normal_distribution<> distrib_real(0.0, std_dev);
	value = distrib_real(gen);
}

weight::~weight()
{
}

void weight::set_front_node(node* front_node)
{
	this->front_node = front_node;
}

node* weight::get_front_node()
{
	return front_node;
}

void weight::set_back_node(node* back_node)
{
	this->back_node = back_node;
}

node* weight::get_back_node()
{
	return back_node;
}

void weight::set_value(const double& value)
{
	this->value = value;
}

double& weight::get_value()
{
	return value;
}

void weight::set_delta(const double& value)
{
	this->delta = value;
}

double& weight::get_delta()
{
	return delta;
}

void weight::set_first_momentum(const double& first_momentum)
{
	this->first_momentum = first_momentum;
}

double& weight::get_first_momentum()
{
	return first_momentum;
}

void weight::set_second_momentum(const double& second_momentum)
{
	this->second_momentum = second_momentum;
}

double& weight::get_second_momentum()
{
	return second_momentum;
}