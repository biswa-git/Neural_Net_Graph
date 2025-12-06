#include<optimizer.hpp>
#include<iostream>
#include<math.h>

optimizer::optimizer() :t(0), learning_rate(0.001)
{
}

optimizer::~optimizer()
{
}

void optimizer::set_learning_rate(double lr)
{
	this->learning_rate= lr;
}

double optimizer::get_learning_rate() const
{
	return this->learning_rate;
}

void optimizer::set_epoch_count(int epoch)
{
	this->t = epoch;
}

int optimizer::get_epoch_count() const
{
	return this->t;
}

basic::basic()
{
	// Basic SGD default learning rate
	set_learning_rate(0.001);
}

basic::~basic()
{
}


void basic::calculate(sequential& model)
{
	double learning_rate = get_learning_rate();
	
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
				if (!front_node) continue;  // Safety check
				
				front_weight->set_value(front_weight->get_value() - learning_rate * (front_node->get_chain().array() * layer_node->get_activation_value().array()).sum());
			}
			if (layer != *layers.begin())
			{
				layer_node->set_bias(layer_node->get_bias() - learning_rate * layer_node->get_chain().sum());
			}
		}
	}
}

momentum::momentum() : momentum_beta(0.9)
{
	// Momentum default learning rate
	set_learning_rate(0.001);
}

momentum::~momentum()
{
}

void momentum::set_momentum_beta(double beta)
{
	// Clamp beta between 0 and 1
	this->momentum_beta = (beta < 0.0) ? 0.0 : (beta > 1.0) ? 1.0 : beta;
}

double momentum::get_momentum_beta() const
{
	return this->momentum_beta;
}

void momentum::calculate(sequential& model)
{
	double learning_rate = get_learning_rate();
	double beta = get_momentum_beta();
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
				if (!front_node) continue;

				auto grad = (front_node->get_chain().array() * layer_node->get_activation_value().array()).sum();

				double delta = beta * front_weight->get_delta() + (1.0 - beta) * grad;
				front_weight->set_delta(delta);
				front_weight->set_value(front_weight->get_value() - learning_rate * front_weight->get_delta());
				
			}
			if (layer != *layers.begin())
			{
				double bias_grad = layer_node->get_chain().sum();
				double delta = beta * layer_node->get_delta() + (1.0 - beta) * bias_grad;
				layer_node->set_delta(delta);
				layer_node->set_bias(layer_node->get_bias() - learning_rate * delta);
			}
		}
	}

}


adam::adam() :beta1(0.9), beta2(0.999), epsilon(1e-8)
{
	set_learning_rate(0.001);
}


adam::~adam()
{
}

void adam::set_beta1(double b)
{
	this->beta1 = (b < 0.0) ? 0.0 : (b > 1.0) ? 1.0 : b;
}

void adam::set_beta2(double b)
{
	this->beta2 = (b < 0.0) ? 0.0 : (b > 1.0) ? 1.0 : b;
}

void adam::set_epsilon(double e)
{
	this->epsilon = (e > 0.0) ? e : 1e-8;
}

double adam::get_beta1() const
{
	return this->beta1;
}

double adam::get_beta2() const
{
	return this->beta2;
}

double adam::get_epsilon() const
{
	return this->epsilon;
}

void adam::calculate(sequential& model)
{

	double learning_rate = get_learning_rate();
	double beta1 = get_beta1();
	double beta2 = get_beta2();
	double eps = get_epsilon();
	
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
				if (!front_node) continue;

				auto grad = (front_node->get_chain().array() * layer_node->get_activation_value().array()).sum();
				front_weight->set_first_momentum(beta1 * front_weight->get_first_momentum() + (1.0 - beta1) * grad);
				front_weight->set_second_momentum(beta2 * front_weight->get_second_momentum() + (1.0 - beta2) * grad * grad);
				
				// Bias correction
				int epoch = get_epoch_count() + 1;
				double bias_correction_1 = 1.0 - pow(beta1, epoch);
				double bias_correction_2 = 1.0 - pow(beta2, epoch);
				
				// Add epsilon to prevent division by zero
				bias_correction_1 = std::max(bias_correction_1, 1e-10);
				bias_correction_2 = std::max(bias_correction_2, 1e-10);
				
				double first_momentum_bias_corrected = front_weight->get_first_momentum() / bias_correction_1;
				double second_momentum_bias_corrected = front_weight->get_second_momentum() / bias_correction_2;
				
				front_weight->set_delta(first_momentum_bias_corrected / (sqrt(second_momentum_bias_corrected) + eps));
				front_weight->set_value(front_weight->get_value() - learning_rate * front_weight->get_delta());

			}
			// Bias update (Adam)
			if (layer != *layers.begin())
			{
				double bias_grad = layer_node->get_chain().sum();

				// Update moments
				layer_node->set_first_momentum(beta1 * layer_node->get_first_momentum() + (1.0 - beta1) * bias_grad);
				layer_node->set_second_momentum(beta2 * layer_node->get_second_momentum() + (1.0 - beta2) * bias_grad * bias_grad);

				int epoch = get_epoch_count() + 1;
				double bias_correction_1 = 1.0 - pow(beta1, epoch);
				double bias_correction_2 = 1.0 - pow(beta2, epoch);
				
				bias_correction_1 = std::max(bias_correction_1, 1e-10);
				bias_correction_2 = std::max(bias_correction_2, 1e-10);

				double bias_first_momentum_bias_corrected = layer_node->get_first_momentum() / bias_correction_1;
				double bias_second_momentum_bias_corrected = layer_node->get_second_momentum() / bias_correction_2;

				double bias_delta = bias_first_momentum_bias_corrected / (sqrt(bias_second_momentum_bias_corrected) + eps);
				layer_node->set_bias(layer_node->get_bias() - learning_rate * bias_delta);
			}
		}
	}
}