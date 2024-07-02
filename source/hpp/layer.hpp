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
private:
	std::vector<node*> nodes;
	layer* back;
	layer* front;
};