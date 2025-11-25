#pragma once
#include<node.hpp>

class weight
{
public:
	weight(node* = nullptr, node* = nullptr);
	~weight();
	void set_front_node(node*);
	node* get_front_node();
	void set_back_node(node*);
	node* get_back_node();
	void set_value(const double&);
	double& get_value();
	void set_delta(const double&);
	double& get_delta();

	void set_first_momentum(const double&);
	double& get_first_momentum();
	void set_second_momentum(const double&);
	double& get_second_momentum();
private:
	node* back_node;
	node* front_node;
	double value;
	double delta;

	//to be replaced by map
	double first_momentum;
	double second_momentum;

};