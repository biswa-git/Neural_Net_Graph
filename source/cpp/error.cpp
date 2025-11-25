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

error::cross_entropy::cross_entropy()
{
}

error::cross_entropy::~cross_entropy()
{
}

Eigen::VectorXd error::cross_entropy::calculate(const Eigen::VectorXd& reference_value, const Eigen::VectorXd& value) const
{
	auto size = reference_value.size();
	auto clamped_value = value.array().max(epsilon).min(1.0 - epsilon);
	
	return -(reference_value.array() * clamped_value.log()) / static_cast<double>(size);
}

Eigen::VectorXd error::cross_entropy::calculate_derivative(const Eigen::VectorXd& reference_value, const Eigen::VectorXd& value) const
{
	auto size = reference_value.size();
	auto clamped_value = value.array().max(epsilon).min(1.0 - epsilon);
	
	return -(reference_value.array() / clamped_value) / static_cast<double>(size);
}

