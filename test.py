import cv2
from model import Model


def preprocess(im):

    im = cv2.resize(im, dsize=(28,28), interpolation = cv2.INTER_CUBIC)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.bitwise_not(im)

    im = im.reshape(1, 28, 28, 1)
    im = im.astype('float32')
    im /= 255
    
    return im


def main():

    model = Model()
    model.load('mnist.h5')

    im = cv2.imread('images/test0.png')
    im = preprocess(im)

    category = model.classify(im)
    print(category)

if __name__ == '__main__':
	main()



