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

    #Starting the video
    camera = cv2.VideoCapture(1)
    grab_image = False
    result = '-'

    while True:
       
        (grabbed, frame) = camera.read() 
        # if the frame could not be grabbed, then we have reached the end of the video
        if not grabbed:
            break

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
            break

        #to grab the image and classify
        if key == ord("a"):
          grab_image = True

     
        if grab_image:
            im = frame[93:387, 173:467]
            im = preprocess(im)
            
            result = model.classify(im)
            grab_image = False


        cv2.rectangle(frame, (170,90), (470,390), (0,255,0), 3)

        # draw the text and timestamp on the frame
        cv2.putText(frame, "Class: {}".format(result),
            (250, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        cv2.imshow("Number Classifer", frame)


    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
