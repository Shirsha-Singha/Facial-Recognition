# plotting photo with detected faces using opencv cascade classifier
from cv2 import imread, imshow, waitKey
from cv2 import destroyAllWindows, CascadeClassifier
from cv2 import rectangle

#load the photograph
pixels = imread('peoples.jpg')

#loading the pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

#performing face detection in image
bboxes = classifier.detectMultiScale(pixels)

# printing bounded box arounfg persons face
for box in bboxes:
	# extract
	x, y, width, height = box
	x2, y2 = x + width, y + height
	# draw a rectangle over the pixels
	rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
#showing the image
imshow('face detection', pixels)
waitKey(0)
destroyAllWindows()
