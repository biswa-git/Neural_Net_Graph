#include<weight.hpp>
#include <random>

weight::weight(node* back_node, node* front_node) :back_node(back_node), front_node(front_node), value(1.0)
{
	back_node->get_front_weights().emplace_back(this);
	front_node->get_back_weights().emplace_back(this);


	std::random_device rd;  // Obtain a random seed from the OS
	std::mt19937 gen(rd()); // Seed the Mersenne Twister engine


	// Generate a random double between 0.0 and 1.0
	std::normal_distribution<> distrib_real(0.0, 1.0);
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
