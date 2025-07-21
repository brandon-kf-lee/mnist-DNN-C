all:
	gcc mnist_train.c  mnist_file.c neural_network.c -lm -o mnist_train
	gcc mnist_inference.c mnist_file.c neural_network.c -lm -o mnist_inference
