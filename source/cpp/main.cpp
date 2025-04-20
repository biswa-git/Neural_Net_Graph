#include<sequential.hpp>
#include<iostream>
#include<fstream>
#include <algorithm> // for std::shuffle
#include <random>
#include <vector>
#include <numeric>
int main()
{
	int size = 400;
	sequential model({ 
		{1},
		{3,activation::RELU},
		{3,activation::SWISH},
		{3,activation::SWISH},
		{1, activation::LINEAR}});

	Eigen::VectorXd x_1(size);
	Eigen::VectorXd y_1(size);
	for (size_t i = 0; i < size; i++)
	{
		x_1[i] = 20.0 * i / (size - 1);
		y_1[i] = sin(x_1[i]/4);
	}

	std::vector< Eigen::VectorXd> x, y;
	x = { x_1 };
	y = { y_1 };

	model.fit(x, y, 10000, 8);

	model.generate_graphviz("graph.dot");


	int length = size;

	Eigen::VectorXd x_in(length);
	for (size_t i = 0; i < length; i++)
	{
		x_in[i] = 20.0 * i / (length - 1);
		
	}

	std::vector< Eigen::VectorXd> x_in_;
	x_in_ = { x_in };
	auto y_out = model.predict(x_in_);
	std::cout << "Input: " << x_in_[0].transpose() << std::endl;
	std::cout << "Output: " << y_out[0].transpose() << std::endl;
	std::ofstream plt_file("output_predict.plt");
	plt_file << "TITLE = \"Output vs. Input\"\n";
	plt_file << "VARIABLES = \"Input\", \"Output_original\", \"Output\"\n";
	plt_file << "ZONE T=\"Output Data\", I=" << length << ", F=POINT\n";
	for (size_t i = 0; i < length; i++)
	{
		plt_file << x_in[i] << " " << y_1[i] << " " << y_out[0][i] << "\n";
	}
	
	return 0;
}
