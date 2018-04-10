from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

for i in x_train:
    plt.imshow(i, cmap=plt.get_cmap('gray'))
    plt.show()