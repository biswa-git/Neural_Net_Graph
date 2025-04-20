#pragma once
#include<weight.hpp>

class layer
{
public:
	layer(const int& = 0);
	~layer();

	std::vector<node*>& get_nodes();
	void set_number_of_nodes(const int&);
	void connect(layer*);
	void set_activation(activation::activation* activation_pointer);
	activation::activation* get_activation() const;

private:
	std::vector<node*> nodes;
	layer* back;
	layer* front;
	activation::activation* activation_pointer;
};
