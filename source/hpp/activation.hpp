#pragma once
#include<Eigen/Core>

namespace activation
{
	enum ACTIVATION
	{
		LINEAR = 0,
		RELU,
		GELU,
		SIGMOID,
		SWISH
	};

	class activation
	{
	public:
		activation();
		virtual ~activation();
		virtual Eigen::VectorXd activate(const Eigen::VectorXd&) = 0;
		virtual Eigen::VectorXd derivative(const Eigen::VectorXd& input) = 0;
	private:

	};

	class linear : public activation
	{
	public:
		linear();
		~linear();
		Eigen::VectorXd activate(const Eigen::VectorXd&);
		Eigen::VectorXd derivative(const Eigen::VectorXd& input);

	private:

	};

	class ReLU : public activation
	{
	public:
		ReLU();
		~ReLU();
		Eigen::VectorXd activate(const Eigen::VectorXd&);
		Eigen::VectorXd derivative(const Eigen::VectorXd& input);

	private:

	};

	class sigmoid : public activation
	{
	public:
		sigmoid();
		~sigmoid();
		Eigen::VectorXd activate(const Eigen::VectorXd&);
		Eigen::VectorXd derivative(const Eigen::VectorXd& input);

	private:

	};

	class swish : public activation
	{
	public:
		swish();
		~swish();
		Eigen::VectorXd activate(const Eigen::VectorXd&);
		Eigen::VectorXd derivative(const Eigen::VectorXd& input);

	private:

	};

}
