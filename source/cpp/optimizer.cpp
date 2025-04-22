#include<optimizer.hpp>
#include<iostream>

optimizer::optimizer()
{
}

optimizer::~optimizer()
{
}

basic::basic()
{
}

basic::~basic()
{
}

void basic::calculate(sequential& model)
{
	double learning_rate = 3e-4;
	
	auto& layers = model.get_layers();
	
	for (auto it = layers.rbegin() ; it != layers.rend(); it++)
	{
		auto& layer = *it;
		auto& layer_nodes = layer->get_nodes();

		for (auto layer_node : layer_nodes)
		{
			for (auto front_weight : layer_node->get_front_weights())
			{
				node* front_node = front_weight->get_front_node();
				front_weight->set_value(front_weight->get_value() - learning_rate * (front_node->get_chain().array() * layer_node->get_activation_value().array()).sum());
			}
			if (layer != *layers.begin())
			{
				layer_node->set_bias(layer_node->get_bias() - learning_rate * layer_node->get_chain().sum());
			}
		}
	}
}

momentum::momentum()
{
}

momentum::~momentum()
{
}

void momentum::calculate(sequential& model)
{
	double learning_rate = 0.1;
	auto& layers = model.get_layers();

	for (auto it = layers.rbegin(); it != layers.rend(); it++)
	{
		auto& layer = *it;
		auto& layer_nodes = layer->get_nodes();

		for (auto layer_node : layer_nodes)
		{
			for (auto front_weight : layer_node->get_front_weights())
			{
				node* front_node = front_weight->get_front_node();
				front_weight->set_value(front_weight->get_value() - learning_rate * (front_node->get_chain().array() * layer_node->get_activation_value().array()).sum());
			}
			if (layer != *layers.begin())
			{
				layer_node->set_bias(layer_node->get_bias() - learning_rate * layer_node->get_chain().sum());
			}
		}
	}

}