#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include "mnist_loader.hpp"

// Minimal example showing how to use MNISTLoader

class MNISTExample
{
public:
	static void runMinimalExample()
	{
		// 1. Create loader
		MNISTLoader loader(28, 28);

		// 2. Load data
		std::cout << "Loading MNIST data..." << std::endl;
		if (!loader.loadTrainingData("D:/Project/Neural_Net_Graph/input/trainingSample")) {
			std::cerr << "Failed to load data!" << std::endl;
			return;
		}

		// 3. Prepare data
		std::cout << "Preparing data..." << std::endl;
		loader.shuffle();
		loader.normalize();

		std::vector<MNISTLoader::Sample> train_samples, test_samples;
		loader.splitData(0.8, train_samples, test_samples);

		std::cout << "Loaded " << train_samples.size() << " training samples" << std::endl;
		std::cout << "Loaded " << test_samples.size() << " test samples" << std::endl;

		// 4. Convert to model input format
		std::vector<Eigen::VectorXd> x_train, y_train;
		loader.convertToModelInput(train_samples, x_train, y_train);

		// 5. Now use x_train and y_train with your sequential model:
		// sequential model({
		//     {784},
		//     {256, activation::RELU},
		//     {128, activation::RELU},
		//     {10, activation::SOFTMAX}
		// });
		// model.fit(x_train, y_train, EPOCHS, BATCH_SIZE);

		std::cout << "\nData ready for training!" << std::endl;
		std::cout << "Input size: " << x_train.size() << " x " << x_train[0].size() << std::endl;
		std::cout << "Output size: " << y_train.size() << " x " << y_train[0].size() << std::endl;
	}

	// Access individual sample data
	static void accessSampleData()
	{
		MNISTLoader loader(28, 28);
		loader.loadTrainingData("D:/Project/Neural_Net_Graph/input/trainingSample");

		const auto& samples = loader.getSamples();
		if (!samples.empty()) {
			std::cout << "\nFirst sample info:" << std::endl;
			std::cout << "Label: " << samples[0].label << std::endl;
			std::cout << "Image size: " << samples[0].image.size() << std::endl;
			std::cout << "First 10 pixel values: ";
			for (int i = 0; i < 10 && i < samples[0].image.size(); ++i) {
				std::cout << samples[0].image[i] << " ";
			}
			std::cout << std::endl;
		}
	}
};
