#include<sequential.hpp>

int main()
{
	sequential model({ {1},  { 1,RELU } });

	Eigen::VectorXd x_1(4);
	Eigen::VectorXd x_2(4);
	Eigen::VectorXd y_1(4);

	x_1 << 1, 2, -3, 4;
	x_2 << 1, 1, 1, 1;
	y_1 << 1, 4, 9, 16;

	std::vector< Eigen::VectorXd> x, y;
	x = { x_1 };
	y = { y_1 };

	model.fit(x, y, 4);

	return 0;

}