import keras
from keras.datasets import mnist

from model import Model


def load_data():
	img_rows, img_cols = 28, 28
	num_classes = 10

	#loading the data
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	print('Training samples :', x_train.shape[0])
	print('Testing samples :', y_test.shape[0])

	#channels last
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

	#preprocessing
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	#one hot vectors
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return x_train, y_train, x_test, y_test



def main():
	x_train, y_train, x_test, y_test = load_data()

	model = Model()
	score = model.train(x_train, y_train, x_test, y_test)

	print('Accuracy =', score)
	model.save('mnist.h5')


if __name__ == '__main__':
	main()







