#include<layer.hpp>

layer::layer(const int& number_of_nodes) :back(nullptr), front(nullptr)
{
	set_number_of_nodes(number_of_nodes);
}

layer::~layer()
{
}

std::vector<node*>& layer::get_nodes()
{
	return nodes;
}

void layer::set_number_of_nodes(const int& number_of_nodes)
{
	for (auto& node_pointer : nodes)
	{
		if (node_pointer != nullptr)
		{
			delete(node_pointer);
		}
	}

	nodes.clear();
	nodes.resize(number_of_nodes);

	for (auto& node_pointer : nodes)
	{
		node_pointer = new node();
	}
}

void layer::connect(layer* next_layer_pointer)
{
	auto current_layer_pointer = this;

	current_layer_pointer->front = next_layer_pointer;
	next_layer_pointer->back = current_layer_pointer;

	for (auto node_pointer_of_current_layer : current_layer_pointer->get_nodes())
	{
		for (auto node_pointer_of_next_layer : next_layer_pointer->get_nodes())
		{
			new weight(node_pointer_of_current_layer, node_pointer_of_next_layer);
		}
	}
}
