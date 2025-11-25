#include<sequential.hpp>
#include<iostream>
#include<fstream>
#include <vector>
#include <iomanip>
#include <cmath>

// Configuration constants
constexpr int RANGE = 21;
constexpr int EPOCHS = 8000;
constexpr int BATCH_SIZE = 16;
constexpr double DOMAIN_MIN = -5.0;
constexpr double DOMAIN_MAX = 5.0;
constexpr double GAUSSIAN_SIGMA = 2.0;
constexpr double GAUSSIAN_AMPLITUDE = 5.0;
constexpr const char* OUTPUT_FILE = "test3d.plt";

double computeGaussian(double x, double y, double sigma, double amplitude)
{
	double exponent = -(x * x + y * y) / (2.0 * sigma * sigma);
	return amplitude * std::exp(exponent);
}

void generateData(int range, double domain_min, double domain_max,
	Eigen::VectorXd& x_1, Eigen::VectorXd& x_2, Eigen::VectorXd& y_1)
{
	int size = range * range;
	x_1.resize(size);
	x_2.resize(size);
	y_1.resize(size);

	for (size_t i = 0; i < range; i++)
	{
		for (size_t j = 0; j < range; j++)
		{
			double normalized_i = i / double(range - 1);
			double normalized_j = j / double(range - 1);

			x_1[i * range + j] = domain_min + (domain_max - domain_min) * normalized_i;
			x_2[i * range + j] = domain_min + (domain_max - domain_min) * normalized_j;
			y_1[i * range + j] = computeGaussian(x_1[i * range + j], x_2[i * range + j],
				GAUSSIAN_SIGMA, GAUSSIAN_AMPLITUDE);
		}
	}
}

int main()
{
	sequential model({
		{2},
		{8, activation::SWISH},
		{12, activation::SWISH},
		{8, activation::SWISH},
		{1, activation::LINEAR},
	});

	// ============================================
	// CHANGE OPTIMIZER HERE
	// ============================================
	// Option 1: Use Basic SGD
	// model.set_optimizer_basic();

	// Option 2: Use Momentum optimizer
	//model.set_optimizer_momentum(0.9);

	// Option 3: Use Adam optimizer (default, already set in constructor)
	model.set_optimizer_adam(0.9, 0.999, 1e-8);
	// ============================================

	Eigen::VectorXd x_1, x_2, y_1;
	generateData(RANGE, DOMAIN_MIN, DOMAIN_MAX, x_1, x_2, y_1);

	std::vector<Eigen::VectorXd> x = {x_1, x_2};
	std::vector<Eigen::VectorXd> y = {y_1};

	model.fit(x, y, EPOCHS, BATCH_SIZE);
	model.generate_graphviz("graph.dot");

	// Generate prediction data
	Eigen::VectorXd x_in_1, x_in_2;
	generateData(RANGE, DOMAIN_MIN, DOMAIN_MAX, x_in_1, x_in_2, y_1);

	std::vector<Eigen::VectorXd> x_in = {x_in_1, x_in_2};
	auto y_out = model.predict(x_in);

	// Write Tecplot output
	std::ofstream fout(OUTPUT_FILE);
	if (!fout) {
		std::cerr << "Failed to open " << OUTPUT_FILE << " for writing.\n";
		return 1;
	}

	fout << "TITLE = \"Synthetic structured data\"\n";
	fout << "VARIABLES = \"X\",\"Y\",\"Z\",\"V\"\n";
	fout << "ZONE I=" << RANGE << ", J=" << RANGE << ", DATAPACKING=POINT\n";
	fout << std::fixed << std::setprecision(8);

	for (int j = 0; j < RANGE; ++j) {
		double y_coord = DOMAIN_MIN + (DOMAIN_MAX - DOMAIN_MIN) * (j / double(RANGE - 1));
		for (int i = 0; i < RANGE; ++i) {
			double x_coord = DOMAIN_MIN + (DOMAIN_MAX - DOMAIN_MIN) * (i / double(RANGE - 1));
			double z = computeGaussian(x_coord, y_coord, GAUSSIAN_SIGMA, GAUSSIAN_AMPLITUDE);
			double V = z - y_out[0][i * RANGE + j];

			fout << x_coord << " " << y_coord << " " << z << " " << V << "\n";
		}
	}

	fout.close();
	std::cout << "Wrote Tecplot file: " << OUTPUT_FILE << " (" << RANGE << "x" << RANGE << " points)\n";
	return 0;
}
