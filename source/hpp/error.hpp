#pragma once
#include<Eigen/Core>

namespace error
{
	class error
	{
	public:
		error();
		~error();
		virtual Eigen::VectorXd calculate(const Eigen::VectorXd&, const Eigen::VectorXd&) const = 0;
		virtual Eigen::VectorXd calculate_derivative(const Eigen::VectorXd&, const Eigen::VectorXd&) const = 0;
	private:

	};

	class mse: public error
	{
	public:
		mse();
		~mse();
		Eigen::VectorXd calculate(const Eigen::VectorXd&, const Eigen::VectorXd&) const;
		Eigen::VectorXd calculate_derivative(const Eigen::VectorXd&, const Eigen::VectorXd&) const;

	private:

	};

	class cross_entropy : public error
	{
	public:
		cross_entropy();
		~cross_entropy();
		Eigen::VectorXd calculate(const Eigen::VectorXd&, const Eigen::VectorXd&) const;
		Eigen::VectorXd calculate_derivative(const Eigen::VectorXd&, const Eigen::VectorXd&) const;

	private:
		double epsilon = 1e-7; // Small constant to prevent log(0)
	};

}
