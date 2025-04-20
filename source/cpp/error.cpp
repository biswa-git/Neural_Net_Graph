#include<error.hpp>

error::error::error()
{
}

error::error::~error()
{
}

error::mse::mse()
{
}

error::mse::~mse()
{
}

Eigen::VectorXd error::mse::calculate(const Eigen::VectorXd& reference_value, const Eigen::VectorXd& value) const
{
	auto size = reference_value.size();
	return 0.5 * (reference_value - value).array().pow(2.0)/static_cast<double>(size);
}

Eigen::VectorXd error::mse::calculate_derivative(const Eigen::VectorXd& reference_value, const Eigen::VectorXd& value) const
{
	auto size = reference_value.size();
	return (reference_value - value).array()/static_cast<double>(size);
}
