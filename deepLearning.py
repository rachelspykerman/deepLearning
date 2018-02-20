import cv2
import numpy as np

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# next capture the video feed
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    frame = cv2.resize(frame,(300,300))
    (h, w) = frame.shape[:2]

    # load the prototxt and caffe model files
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    # preprocess the image (mean subtraction, and no scaling)
    # we shape to 300x300 because inside prototxt, 300x300 is specified
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)

    # feed the blob into the deep learning neural network and get the predicted probabilities as an output
    # Note at this step that we aren’t training a CNN, but we are making use of a pre-trained network. Therefore we are
    # just passing the blob through the network (i.e., forward propagation) to obtain the result (no back-propagation).
    net.setInput(blob)
    prediction = net.forward()
    # the 3rd value in the shape tells how many items were detected, and values 3 to 7 are the location
    # of the object in the image (it's a % of the width and height of the image)
    print(prediction.shape)

    # loop over the detections
    for i in np.arange(0, prediction.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        print("Prediction is {}".format(prediction)) # each new array in prediction is for each diff object detected
        confidence = prediction[0, 0, i, 2]
        idx1 = prediction[0,0,i,1]
        print(idx1)  # this is the index in the CLASSES array, example if idx1 = 2, then CLASSES[2] = bicycle
        print(prediction.shape[2])
        print(confidence)

        ''' Two objects detected, person 99% and sofa 39%
        print(prediction.shape)
        print(prediction)
        print(prediction[0,0,i,1])
        print(prediction.shape[2])
        print(prediction[0,0,i,2])
        
        (1, 1, 2, 7)
        [[[[0.0000000e+00 1.5000000e+01 9.9548572e-01 2.1319434e-01
            2.2259659e-01 7.9909253e-01 1.0147104e+00]
           [0.0000000e+00 1.8000000e+01 3.9246237e-01 5.6102276e-03
            6.9613898e-01 1.4996165e-01 9.9121165e-01]]]]
        15.0
        2
        0.9954857
        
        [[[[0.0000000e+00 1.5000000e+01 9.9548572e-01 2.1319434e-01
            2.2259659e-01 7.9909253e-01 1.0147104e+00]
           [0.0000000e+00 1.8000000e+01 3.9246237e-01 5.6102276e-03
            6.9613898e-01 1.4996165e-01 9.9121165e-01]]]]
        18.0
        2
        0.39246237'''

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.2:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(prediction[0, 0, i, 1])
            box = prediction[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output image
    cv2.imshow("Image", frame)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release the video capture
camera.release()
cv2.destroyAllWindows()



# To get correct predictions from deep neural networks we need to preprocess the data. This usually involves
# mean subtraction and scaling by some factor. The dnn module contains the blobFromImage(s) functions that can
# be used for pre-processing images and preparing them for classification via pre-training deep learning models

# MEAN SUBTRACTION
# Mean subtraction is where we find the mean/average pixel value for each color channel (RGB) and subtract that
# from the original image. This helps combat illumination changes in the input images in the dataset

# SCALING
# If you want to normalize your data, you can divide the average RGB values by a sigma value. The value of sigma
# may be the standard deviation across the training set (thereby turning the pre-processing step into a standard
# score/z-score). However, sigma may also be manually set (versus calculated) to scale the input image space
# into a particular range. Or it may not be set at all, it's up to the architect

# mu_R, mu_G, mu_B are the average pixel intensities for each color channel
# R = (R - mu_R)/sigma
# G = (G - mu_G)/sigma
# B = (B - mu_B)/sigma

# BLOBS
# A blob is just a collection of image(s) with the same spatial dimensions (width and height), same depth (number
# of channels), that have all be preprocessed in the same manner. So in the dnn library, we use blobFromImage(s)
# to perform mean subtraction, normalization, and/or channel swapping on the image dataset before feeding images
# into the neural network

# Note blobFromImages using the exact same syntax as below
# blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)

# image : Input image that is to be preprocessed before passing it through our deep neural network for classification.
# scalefactor : After mean subtraction is performed, we can optionally scale our images by some factor. This value
#               defaults to 1.0 (i.e., no scaling). Note that scalefactor should be 1/sigma as we’re actually
#               multiplying the input channels (after mean subtraction) by scalefactor .
# size : Supply the spatial size that the Convolutional Neural Network expects. For most current state-of-the-art
#        neural networks this is either 224×224, 227×227, or 299×299.
# mean : These are the mean subtraction values. They can be a 3-tuple of the RGB means or they can be a single value
#        in which case the supplied value is subtracted from every channel of the image. Note, ensure tuple is of
#        the form (R,G,B), not (B,G,R)
# swapRB : OpenCV assumes images are in BGR channel order; however, the mean value assumes we are using RGB order.
#          To resolve this discrepancy we can swap the R and B channels in image  by setting this value to True.
#          By default OpenCV performs this channel swapping for us.
# The cv2.dnn.blobFromImage function returns a blob which is our input image after mean subtraction, normalizing, and
# channel swapping.
