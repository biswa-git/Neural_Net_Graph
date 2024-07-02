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
private:
	node* back_node;
	node* front_node;
	double value;
};