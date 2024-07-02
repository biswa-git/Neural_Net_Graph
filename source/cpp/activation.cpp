#include<Activation.hpp>
#include<cmath>
/*
Eigen::VectorXd SigmoidActivation::activate(const Eigen::VectorXd& input) const
{
    return 1.0 / (1.0 + (-input.array()).exp());
}


Eigen::VectorXd SigmoidActivation::derivative(const Eigen::VectorXd& input) const
{
    Eigen::VectorXd sigmoid = activate(input);
    return sigmoid.array() * (1.0 - sigmoid.array());
}


Eigen::VectorXd ReLUActivation::activate(const Eigen::VectorXd& input) const
{
    return input.cwiseMax(0.0);
}

Eigen::VectorXd ReLUActivation::derivative(const Eigen::VectorXd& input) const
{
    return (input.array() > 0.0).select(Eigen::VectorXd::Ones(input.size()), 0.0);
}


Eigen::VectorXd LinearActivation::activate(const Eigen::VectorXd& input) const
{
    return input;
}

Eigen::VectorXd LinearActivation::derivative(const Eigen::VectorXd& input) const
{
    return Eigen::VectorXd::Ones(input.size());
}



Eigen::VectorXd GELUActivation::activate(const Eigen::VectorXd& input) const
{
    return 0.5 * input.array() * (1.0 + tanh(sqrt(2.0 / M_PI) * (input.array() + 0.044715 * input.array().pow(3))));
}

Eigen::VectorXd GELUActivation::derivative(const Eigen::VectorXd& input) const
{
    Eigen::VectorXd cdf = 0.5 * (1.0 + tanh(sqrt(2.0 / M_PI) * (input.array() + 0.044715 * input.array().pow(3))));
    return 0.5 + 0.5 * cdf.array() * (1.0 - cdf.array());
}


Eigen::VectorXd UnitActivation::activate(const Eigen::VectorXd& input) const
{
    return Eigen::VectorXd::Ones(input.size());
}

Eigen::VectorXd UnitActivation::derivative(const Eigen::VectorXd& input) const
{
    return Eigen::VectorXd::Zero(input.size());
}

Eigen::VectorXd SoftmaxActivation::activate(const Eigen::VectorXd& input) const
{
    Eigen::VectorXd expInput = input.array().exp();
    return expInput / expInput.sum();
}

Eigen::VectorXd SoftmaxActivation::derivative(const Eigen::VectorXd& input) const
{
    // Derivative of softmax is not implemented, as it requires Jacobian matrix
    throw std::runtime_error("Softmax derivative is not directly implemented. Use softmax with cross-entropy loss for training.");
}
*/

activation::activation()
{
}

activation::~activation()
{
}

linear_activation::linear_activation()
{
}

linear_activation::~linear_activation()
{
}

Eigen::VectorXd linear_activation::calculate(const Eigen::VectorXd& input)
{
	return input;
}

Eigen::VectorXd linear_activation::calculate_derivative(const Eigen::VectorXd& input)
{
    return Eigen::VectorXd::Ones(input.size());
}

ReLU_activation::ReLU_activation()
{
}

ReLU_activation::~ReLU_activation()
{
}

Eigen::VectorXd ReLU_activation::calculate(const Eigen::VectorXd& input)
{
    return input.cwiseMax(0.0);
}

Eigen::VectorXd ReLU_activation::calculate_derivative(const Eigen::VectorXd& input)
{
    return (input.array() > 0.0).select(Eigen::VectorXd::Ones(input.size()), 0.0);
}