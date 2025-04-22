#include<sequential.hpp>
#include<iostream>
#include<fstream>
#include <random>
#include <numeric>


sequential::sequential(const std::vector<std::vector<int>>& inputs) :batch_size(0), error(new error::mse()), opt(new basic())
{
	for (auto& input : inputs)
	{
		auto new_layer = new layer(input[0]);

		if (input.size() > 1)
		{
			auto& new_layer_nodes = new_layer->get_nodes();

			switch (input[1])
			{
			case activation::LINEAR:
				new_layer->set_activation(new activation::linear());
				break;
			case activation::RELU:
				new_layer->set_activation(new activation::ReLU());
				break;
			case activation::SIGMOID:
				new_layer->set_activation(new activation::sigmoid());
				break;
			case activation::SWISH:
				new_layer->set_activation(new activation::swish());
				break;
			default:
				// invalid activation  
				break;
			}

		}
		else
		{
			//invalid layer
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

std::vector<layer*>& sequential::get_layers()
{
	return layers;
}

void sequential::fit(const std::vector< Eigen::VectorXd>& x, const std::vector< Eigen::VectorXd>& y, const int& epoch, const int& batch_size)
{
	// Let get the dimesion of input and output
	auto input_layer = layers.begin();
	auto output_layer = layers.rbegin();

	auto& input_layer_nodes = (*input_layer)->get_nodes();
	auto& output_layer_nodes = (*output_layer)->get_nodes();

	const auto input_dim = input_layer_nodes.size();

	this->batch_size = batch_size;
	this->x.resize(input_dim);
	this->y.resize(output_layer_nodes.size());

	const auto sample_size = static_cast<int>(x[0].size());

	std::random_device rd;
	std::mt19937 g(rd());
	std::vector<int> indices(sample_size);
	std::iota(indices.begin(), indices.end(), 0);

	//decide starting and end indices of batch
	const int num_of_batch = sample_size / batch_size;
	
	std::vector<std::pair<int, int>> indices_start_end_pairs;
	for (size_t i = 0; i < num_of_batch; i++)
	{
		indices_start_end_pairs.push_back(std::pair<int, int>(i * batch_size, (i + 1) * batch_size - 1));
	}

	if (sample_size % batch_size != 0)
	{
		indices_start_end_pairs.push_back(std::pair<int, int>(num_of_batch * batch_size, sample_size - 1));
	}

	for (size_t i_epoch = 0; i_epoch < epoch; i_epoch++)
	{
		std::shuffle(indices.begin(), indices.end(), g);
		
		for (auto& indices_start_end_pair : indices_start_end_pairs)
		{
			this->batch_size = indices_start_end_pair.second - indices_start_end_pair.first + 1;

			for (size_t i_input_dim = 0; i_input_dim < input_dim; i_input_dim++)
			{
				this->x[i_input_dim].resize(this->batch_size);
				this->y[i_input_dim].resize(this->batch_size);

				for (int i_batch_size = 0; i_batch_size < this->batch_size; ++i_batch_size)
				{
					this->x[i_input_dim][i_batch_size] = x[i_input_dim][indices[i_batch_size]];
					this->y[i_input_dim][i_batch_size] = y[i_input_dim][indices[i_batch_size]];
				}

				input_layer_nodes[i_input_dim]->set_activation_value(this->x[i_input_dim]);
			}

			forward_pass();
			backpropagate();

		}

		
		//calculate error
		auto& output_layer = layers.rbegin();
		auto& output_layer_nodes = (*output_layer)->get_nodes();

		double total_error = 0;
		for (int i = 0; i < output_layer_nodes.size(); i++)
		{
			total_error += error->calculate(this->y[i], output_layer_nodes[i]->get_activation_value()).sum();
		}

		if (i_epoch%1000 == 0)
		{
			std::cout << "error after " << i_epoch << " epoch  = " << total_error << std::endl;
			std::cout << "--------------------------------------------------------" << std::endl;
		}
	}

}

void sequential::initialize()
{

}

void sequential::forward_pass()
{
	for (auto it = layers.begin() + 1; it != layers.end(); it++)
	{
		auto& layer = *it;
		auto& layer_nodes = layer->get_nodes();
		for (auto layer_node : layer_nodes)
		{
			auto& back_weights = layer_node->get_back_weights();
			Eigen::VectorXd sum = layer_node->get_bias() * Eigen::VectorXd::Ones(batch_size);

			for (auto back_weight : back_weights)
			{
				sum += back_weight->get_value() * back_weight->get_back_node()->get_activation_value();
			}

			//layer_node->set_value(sum);
			layer_node->set_activation_value(layer_node->get_activation()->activate(sum));
			layer_node->set_derivative_value(layer_node->get_activation()->derivative(sum));
		}
	}
}

void sequential::backpropagate()
{
	double learning_rate = 1e-2;
	auto& output_layer = layers.rbegin();
	auto& output_layer_nodes = (*output_layer)->get_nodes();

	for (int i = 0; i < output_layer_nodes.size(); i++)
	{
		auto layer_node = output_layer_nodes[i];
		layer_node->set_chain(error->calculate_derivative(layer_node->get_activation_value(), y[i]).array() * layer_node->get_derivative_value().array());

		//layer_node->set_bias(layer_node->get_bias() - learning_rate * layer_node->get_chain().sum());
	}

	for (auto it = layers.rbegin() + 1; it != layers.rend(); it++)
	{
		auto& layer = *it;
		auto& layer_nodes = layer->get_nodes();

		for (auto layer_node : layer_nodes)
		{
			layer_node->set_chain(Eigen::VectorXd::Zero(batch_size));

			for (auto front_weight : layer_node->get_front_weights())
			{
				node* front_node = front_weight->get_front_node();
				layer_node->set_chain(layer_node->get_chain().array() + front_node->get_chain().array() * front_weight->get_value());
				//front_weight->set_value(front_weight->get_value() - learning_rate * (front_node->get_chain().array() * layer_node->get_value().array()).sum());
			    //dont update the weight. save the gradient
			}
			if (layer != *layers.begin())
			{
				layer_node->set_chain(layer_node->get_chain().array() * layer_node->get_derivative_value().array());

				//layer_node->set_bias(layer_node->get_bias() - learning_rate * layer_node->get_chain().sum());
				//dont update the bias. save the gradient
			}
		}
	}

	opt->calculate(*this);
}




void sequential::print_network() const {
	std::cout << "Network Structure:" << std::endl;
	for (size_t i = 0; i < layers.size(); ++i) {
		std::cout << "Layer " << i + 1 << ":" << std::endl;
		const auto& nodes = layers[i]->get_nodes();
		for (size_t j = 0; j < nodes.size(); ++j) {
			std::cout << "  Node " << j + 1 << " - Value: " << nodes[j]->get_activation_value().transpose() << std::endl;
		}
	}
}

void sequential::generate_graphviz(const std::string& filename) const {
	std::ofstream file(filename);
	file << "digraph G {\n";
	file << "  rankdir=LR;\n"; // Set the direction from right to left
	file << "  graph [splines=true, nodesep=1, ranksep=2];\n";
	file << "  node [shape=ellipse, style=filled, color=lightblue, fontname=\"Helvetica\", fontsize=10];\n";
	file << "  edge [color=gray, arrowhead=vee, arrowsize=0.7];\n";

	for (size_t i = 0; i < layers.size(); ++i) {
		const auto& nodes = layers[i]->get_nodes();
		for (size_t j = 0; j < nodes.size(); ++j) {
			file << "  node" << i << "_" << j << " [label=\"Layer " << i + 1 << " Node " << j + 1 << "\\nBias: " << nodes[j]->get_bias() << "\"];\n";
		}
	}

	for (size_t i = 0; i < layers.size() - 1; ++i) {
		const auto& current_layer_nodes = layers[i]->get_nodes();
		const auto& next_layer_nodes = layers[i + 1]->get_nodes();
		for (size_t j = 0; j < current_layer_nodes.size(); ++j) {
			for (size_t k = 0; k < next_layer_nodes.size(); ++k) {
				const auto& back_weights = next_layer_nodes[k]->get_back_weights();
				for (const auto& back_weight : back_weights) {
					if (back_weight->get_back_node() == current_layer_nodes[j]) {
						file << "  node" << i << "_" << j << " -> node" << i + 1 << "_" << k << " [label=\"Weight: " << back_weight->get_value() << "\"];\n";
					}
				}
			}
		}
	}

	file << "}\n";
	file.close();
}


const std::vector<Eigen::VectorXd> sequential::predict(const std::vector<Eigen::VectorXd>& x) const
{
	// Set the input values
	auto input_layer = layers.begin();
	auto& input_layer_nodes = (*input_layer)->get_nodes();
	for (int i = 0; i < input_layer_nodes.size(); i++)
	{
		input_layer_nodes[i]->set_activation_value(x[i]);

		std::cout << "Input: " << input_layer_nodes[i]->get_activation_value() << std::endl;
	}
	// Perform forward pass
	for (auto it = layers.begin() + 1; it != layers.end(); it++)
	{
		auto& layer = *it;
		auto& layer_nodes = layer->get_nodes();
		for (auto layer_node : layer_nodes)
		{
			auto& back_weights = layer_node->get_back_weights();
			Eigen::VectorXd sum = layer_node->get_bias() * Eigen::VectorXd::Ones(x[0].size()); // batch_size is 1 for prediction

			for (auto back_weight : back_weights)
			{
				sum += back_weight->get_value() * back_weight->get_back_node()->get_activation_value();
			}

			//layer_node->set_value(sum);
			layer_node->set_activation_value(layer_node->get_activation()->activate(sum));
		}
	}

	// Get the output values
	auto& output_layer = layers.back();
	auto& output_layer_nodes = output_layer->get_nodes();
	std::vector<Eigen::VectorXd> output;
	for (auto layer_node : output_layer_nodes)
	{
		output.push_back(layer_node->get_activation_value());
	}

	return output;
}

