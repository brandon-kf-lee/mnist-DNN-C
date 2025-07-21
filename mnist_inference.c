/* 
 * Glue code that uses pre-trained weights and reports algorithm accuracy
 * Author: Brandon Lee, brandon.kf.lee@gmail.com
 *     Derived from: Andrew Carter, https://github.com/AndrewCarterUK/mnist-neural-network-plain-c
 */ 

#include "include/mnist_file.h"
#include "include/neural_network.h"

/**
 * Downloaded from: http://yann.lecun.com/exdb/mnist/
 */
const char * test_images_file = "data/t10k-images-idx3-ubyte";
const char * test_labels_file = "data/t10k-labels-idx1-ubyte";

/**
 * Calculate the accuracy of the predictions of a neural network on a dataset.
 */
float calculate_accuracy(mnist_dataset_t * dataset, neural_network_t * network)
{
    float activations[MNIST_LABELS], max_activation;
    int i, j, correct, predict;

    // Loop through the dataset
    for (i = 0, correct = 0; i < dataset->size; i++) {
        // Calculate the activations for each image using the neural network
        neural_network_hypothesis(&dataset->images[i], network, activations);

        // Set predict to the index of the greatest activation
        for (j = 0, predict = 0, max_activation = activations[0]; j < MNIST_LABELS; j++) {
            if (max_activation < activations[j]) {
                max_activation = activations[j];
                predict = j;
            }
        }

        // Increment the correct count if we predicted the right label
        if (predict == dataset->labels[i]) {
            correct++;
        }
    }

    // Return the percentage we predicted correctly as the accuracy
    return ((float) correct) / ((float) dataset->size);
}

int main(int argc, char *argv[]){

    mnist_dataset_t * test_dataset;
    neural_network_t network;

    // Read the datasets from the files
    test_dataset = mnist_get_dataset(test_images_file, test_labels_file);

    // Load the pre-trained network
    FILE * mnist_network = fopen("mnist_network", "rb");
    fread(&network, sizeof(network), 1, mnist_network);
    fclose(mnist_network);

    // Calculate the accuracy using the test dataset
    float accuracy = calculate_accuracy(test_dataset, &network);
    printf("Accuracy: %.3f\n", accuracy);

    //Cleanup
    mnist_free_dataset(test_dataset);

    return 0;
}