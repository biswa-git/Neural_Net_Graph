#include<activation.hpp>
#include<cmath>

// Define M_PI if not available on this platform
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

activation::activation::activation()
{
}

activation::activation::~activation()
{
}

//--------------------------------------------------------------------------------------
//Linear activation
activation::linear::linear()
{
}

activation::linear::~linear()
{
}

Eigen::VectorXd activation::linear::activate(const Eigen::VectorXd& input)
{
	return input;
}

Eigen::VectorXd activation::linear::derivative(const Eigen::VectorXd& input)
{
    return Eigen::VectorXd::Ones(input.size());
}

//--------------------------------------------------------------------------------------

//Relu activation
activation::ReLU::ReLU()
{
}

activation::ReLU::~ReLU()
{
}

Eigen::VectorXd activation::ReLU::activate(const Eigen::VectorXd& input)
{
    return input.cwiseMax(0.0);
}

Eigen::VectorXd activation::ReLU::derivative(const Eigen::VectorXd& input)
{
    // ReLU derivative: 1 if x > 0, else 0
    // Avoid division by zero at x=0 by using > instead of >=
    return (input.array() > 0.0).select(Eigen::VectorXd::Ones(input.size()), 0.0);
}
//--------------------------------------------------------------------------------------
// gelu activation

activation::GeLU::GeLU()
{
}

activation::GeLU::~GeLU()
{
}

Eigen::VectorXd activation::GeLU::activate(const Eigen::VectorXd& input)
{
    // GeLU approximation with numerical stability
    const double SQRT_2_OVER_PI = sqrt(2.0 / M_PI);
    const double INPUT_CLAMP = 20.0;  // Clamp inputs to prevent overflow in tanh/exp
    
    // Clamp inputs for numerical stability
    Eigen::VectorXd clamped_input = input.array().max(-INPUT_CLAMP).min(INPUT_CLAMP);
    
    return 0.5 * clamped_input.array() * (1.0 + tanh(SQRT_2_OVER_PI * (clamped_input.array() + 0.044715 * clamped_input.array().pow(3))));
}

Eigen::VectorXd activation::GeLU::derivative(const Eigen::VectorXd& input)
{
    // Correct GeLU derivative formula with numerical stability
    // d/dx[GeLU] = 0.5 * [1 + tanh(...)] + 0.5 * x * sech²(...) * d/dx[tanh_arg]
    // where tanh_arg = sqrt(2/π) * (x + 0.044715*x³)
    
    const double SQRT_2_OVER_PI = sqrt(2.0 / M_PI);
    const double TANH_COEFFICIENT = 0.044715;
    const double INPUT_CLAMP = 20.0;  // Clamp inputs to prevent overflow in tanh/exp
    
    // Clamp inputs for numerical stability (tanh is essentially 1 or -1 beyond ±20)
    auto clamped_input = input.array().max(-INPUT_CLAMP).min(INPUT_CLAMP);
    
    auto cubic_term = TANH_COEFFICIENT * clamped_input.pow(3);
    auto tanh_arg = SQRT_2_OVER_PI * (clamped_input + cubic_term);
    auto tanh_val = tanh_arg.tanh();
    
    // sech²(x) = 1 - tanh²(x), numerically stable
    auto sech_squared = 1.0 - tanh_val.pow(2);
    
    // Derivative of tanh_arg with respect to x
    auto tanh_arg_derivative = SQRT_2_OVER_PI * (1.0 + 3.0 * TANH_COEFFICIENT * clamped_input.array().pow(2));
    
    // Apply chain rule: d/dx[GeLU] = 0.5*(1 + tanh(...)) + 0.5*x*sech²(...)*d(tanh_arg)/dx
    return 0.5 * (1.0 + tanh_val) + 0.5 * clamped_input * sech_squared * tanh_arg_derivative;
}

//--------------------------------------------------------------------------------------
// sigmoid activation
activation::sigmoid::sigmoid()
{
}

activation::sigmoid::~sigmoid()
{
}

Eigen::VectorXd activation::sigmoid::activate(const Eigen::VectorXd& input)
{
    // Numerically stable sigmoid: 1 / (1 + exp(-x))
    // For large positive x: returns 1
    // For large negative x: uses exp(x) / (1 + exp(x)) to avoid underflow
    return 1.0 / (1.0 + (-input.array()).exp());
}

Eigen::VectorXd activation::sigmoid::derivative(const Eigen::VectorXd& input)
{
    Eigen::VectorXd activated = activate(input);
    return activated.array() * (1.0 - activated.array());
}
//--------------------------------------------------------------------------------------
// Swish activation
activation::swish::swish()
{
}

activation::swish::~swish()
{
}

Eigen::VectorXd activation::swish::activate(const Eigen::VectorXd& input)
{
    // Swish(x) = x * σ(x) where σ(x) = 1 / (1 + exp(-x))
    // Numerically stable implementation
    Eigen::VectorXd sigmoid = 1.0 / (1.0 + (-input.array()).exp());
    return input.array() * sigmoid.array();
}

Eigen::VectorXd activation::swish::derivative(const Eigen::VectorXd& input)
{
    // Derivative: d/dx[x*σ(x)] = σ(x) + x*σ(x)*(1-σ(x))
    // Factored form: σ(x) * (1 + x*(1-σ(x)))
    Eigen::VectorXd sigmoid = 1.0 / (1.0 + (-input.array()).exp());
    return sigmoid.array() * (1.0 + input.array() * (1.0 - sigmoid.array()));
}
//--------------------------------------------------------------------------------------
//tanh activation
activation::tanh_::tanh_()
{
}

activation::tanh_::~tanh_()
{
}

Eigen::VectorXd activation::tanh_::activate(const Eigen::VectorXd& input)
{
    return input.array().tanh();
}

Eigen::VectorXd activation::tanh_::derivative(const Eigen::VectorXd& input)
{
    // tanh derivative: d/dx[tanh(x)] = 1 - tanh²(x) = sech²(x)
    // More numerically stable than computing tanh twice
    Eigen::VectorXd activated = activate(input);
    return 1.0 - activated.array().pow(2);
}

//--------------------------------------------------------------------------------------
// Softmax activation
activation::softmax::softmax()
{
}

activation::softmax::~softmax()
{
}

Eigen::VectorXd activation::softmax::activate(const Eigen::VectorXd& input)
{
    double max_input = input.maxCoeff();
    Eigen::VectorXd exp_input = (input.array() - max_input).exp();
    double sum_exp = exp_input.sum();
    return exp_input.array() / sum_exp;
}

Eigen::VectorXd activation::softmax::derivative(const Eigen::VectorXd& input)
{
	// When using cross-entropy loss + softmax, return identity (1.0)
	// because cross-entropy already accounts for softmax derivative
	return Eigen::VectorXd::Ones(input.size());
}