#include<sequential.hpp>
#include<iostream>

sequential::sequential(const std::vector<std::vector<int>>& inputs) :batch_size(1)
{
	for (auto& input : inputs)
	{
		auto new_layer = new layer(input[0]);

		if (input.size() > 1)
		{
			auto& new_layer_nodes = new_layer->get_nodes();

			if (input[1] == LINEAR)
			{

				for (auto new_layer_node : new_layer_nodes)
				{
					new_layer_node->set_activation(&linear);
				}
			}
			else if (input[1] == RELU)
			{
				for (auto new_layer_node : new_layer_nodes)
				{
					new_layer_node->set_activation(&ReLU);
				}
			}
		}
		else
		{
			auto& new_layer_nodes = new_layer->get_nodes();
			for (auto new_layer_node : new_layer_nodes)
			{
				new_layer_node->set_activation(&linear);
			}
		}
		layers.emplace_back(new_layer);
	}

	layer* prev_layer = nullptr;
	bool input_layer = true;

	for (auto layer : layers)
	{
		if (input_layer)
		{
			input_layer = false;
		}
		else
		{
			prev_layer->connect(layer);
		}
		prev_layer = layer;
	}
}

sequential::~sequential()
{
}

linear_activation sequential::linear;
ReLU_activation sequential::ReLU;

void sequential::fit(const std::vector< Eigen::VectorXd>& x, const std::vector< Eigen::VectorXd>& y, const int& batch_size)
{
	this->x = x;
	this->y = y;
	this->batch_size = batch_size;

	for (auto layer : layers)
	{
		auto& layer_nodes = layer->get_nodes();
		for (auto layer_node : layer_nodes)
		{
			layer_node->get_value() = Eigen::VectorXd::Zero(batch_size);
		}
	}

	auto input_layer = layers.begin();
	auto& input_layer_nodes = (*input_layer)->get_nodes();

	//setting input
	for (int i = 0; i < input_layer_nodes.size(); i++)
	{
		input_layer_nodes[i]->set_value(x[i]);
	}

	//forward pass
	//----------------------------------------------------
	for (auto it = layers.begin() + 1; it != layers.end(); it++)
	{
		auto layer = *it;
		auto& layer_nodes = layer->get_nodes();
		for (auto layer_node : layer_nodes)
		{
			auto& back_weights = layer_node->get_back_weights();
			Eigen::VectorXd sum = layer_node->get_bias() * Eigen::VectorXd::Ones(batch_size);

			for (auto back_weight : back_weights)
			{
				sum += back_weight->get_value() * back_weight->get_back_node()->get_value();
			}

			layer_node->set_value(sum);
			layer_node->set_activation_value(layer_node->get_activation()->calculate(sum));
			layer_node->set_derivative_value(layer_node->get_activation()->calculate_derivative(sum));
		}
	}

	//calculate error
	auto output_layer1 = layers.rbegin();
	auto& output_layer_nodes1 = (*output_layer1)->get_nodes();

	std::vector<double> error(output_layer_nodes1.size());

	for (int i = 0; i < output_layer_nodes1.size(); i++)
	{
		error[i] = 0.5*((output_layer_nodes1[i]->get_activation_value() - y[i]).dot(output_layer_nodes1[i]->get_activation_value() - y[i]));
	}




	/*
	//printing node data
	auto output_layer = layers.rbegin();
	auto& output_layer_nodes = (*output_layer)->get_nodes();
	for (auto output_layer_node:output_layer_nodes)
	{
		std::cout << output_layer_node->get_value() << std::endl;
		std::cout << y[0] << std::endl;
	}
	*/
}
