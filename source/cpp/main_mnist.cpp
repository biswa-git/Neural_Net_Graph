/*#include<sequential.hpp>
#include<mnist_loader.hpp>
#include<iostream>
#include<fstream>
#include <vector>
#include <iomanip>
#include <cmath>

// MNIST Configuration
constexpr int EPOCHS = 5000;
constexpr int BATCH_SIZE = 32;
constexpr double TRAIN_TEST_RATIO = 0.8;
constexpr const char* MNIST_DATA_PATH = "D:/Project/Neural_Net_Graph/input/trainingSample";

int main()
{
	// ============================================
	// LOAD MNIST DATA
	// ============================================
	std::cout << "Loading MNIST data from: " << MNIST_DATA_PATH << std::endl;
	
	MNISTLoader loader(28, 28); // Standard MNIST image size
	if (!loader.loadTrainingData(MNIST_DATA_PATH)) {
		std::cerr << "Failed to load MNIST data!" << std::endl;
		return 1;
	}

	std::cout << "Total samples loaded: " << loader.getSampleCount() << std::endl;

	// Shuffle the data
	loader.shuffle();

	// Normalize pixel values
	loader.normalize();

	// Split into training and test sets
	std::vector<MNISTLoader::Sample> train_samples, test_samples;
	loader.splitData(TRAIN_TEST_RATIO, train_samples, test_samples);

	std::cout << "Training samples: " << train_samples.size() << std::endl;
	std::cout << "Test samples: " << test_samples.size() << std::endl;

	// Convert to model input format
	std::vector<Eigen::VectorXd> x_train, y_train;
	std::vector<Eigen::VectorXd> x_test, y_test;

	loader.convertToModelInput(train_samples, x_train, y_train);
	loader.convertToModelInput(test_samples, x_test, y_test);

	std::cout << "Input dimensions: " << x_train.size() << " (784 pixels)" << std::endl;
	std::cout << "Output dimensions: " << y_train.size() << " (10 classes)" << std::endl;

	// ============================================
	// CREATE AND CONFIGURE NEURAL NETWORK
	// ============================================
	sequential model({
		{784},                              // Input layer: 28x28 = 784 pixels
		{256, activation::SWISH},            // Hidden layer 1: 256 neurons with ReLU
		{128, activation::SWISH},            // Hidden layer 2: 128 neurons with ReLU
		{64, activation::SWISH},             // Hidden layer 3: 64 neurons with ReLU
		{10, activation::SOFTMAX},          // Output layer: 10 neurons for digits 0-9 with Softmax
	});

	// Set optimizer
	model.set_optimizer_adam(0.9, 0.999, 1e-8);

	// Set learning rate (higher than default 0.001 for MNIST training)
	model.set_learning_rate(0.01);

	// ============================================
	// TRAIN THE MODEL
	// ============================================
	std::cout << "\n=== Starting Training ===" << std::endl;

	model.fit(x_train, y_train, EPOCHS, BATCH_SIZE);
	// ============================================
	// GENERATE NETWORK VISUALIZATION
	// ============================================
	model.generate_graphviz("mnist_network.dot");
	std::cout << "Network visualization saved to: mnist_network.dot" << std::endl;

	// ============================================
	// EVALUATE ON TEST SET
	// ============================================
	std::cout << "\n=== Evaluating on Test Set ===" << std::endl;
	
	auto predictions = model.predict(x_test);
	int correct = 0;
	int total = test_samples.size();

	for (size_t i = 0; i < total; ++i) {
		// Find the predicted digit (argmax of output)
		int predicted_digit = 0;
		double max_output = predictions[0][i];

		for (int j = 1; j < 10; ++j) {
			if (predictions[j][i] > max_output) {
				max_output = predictions[j][i];
				predicted_digit = j;
			}
		}

		// Get the true digit from one-hot encoded output
		int true_digit = test_samples[i].label;

		if (predicted_digit == true_digit) {
			correct++;
		}

		// Print first 10 predictions
		if (i < 10) {
			std::cout << "Sample " << i << ": Predicted=" << predicted_digit 
				      << " (confidence: " << std::fixed << std::setprecision(2) << max_output * 100 << "%)"
					  << ", True=" << true_digit;
			if (predicted_digit == true_digit) {
				std::cout << " ?";
			}
			else {
				std::cout << " ?";
			}
			std::cout << std::endl;
		}
	}

	double accuracy = (static_cast<double>(correct) / total) * 100.0;
	std::cout << "\nTest Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "% (" 
		      << correct << "/" << total << ")" << std::endl;

	return 0;
}
*/
