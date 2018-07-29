import cv2
import matplotlib.pyplot as plt 
import time

def convertToRGB(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Reads Image and converts it to grayscale	
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
test1 = cv2.imread('/home/racs/DATA/Projects/FaceDetection/Images/test4.jpg')
grayImg = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

choice = input("1.Haar 2.LBP")
if(choice == 1):
	#load haar classifier
	haarFaceCascade = cv2.CascadeClassifier('/home/racs/DATA/Projects/FaceDetection/xmlFiles/haarcascade_frontalface_alt.xml')
	faces = haarFaceCascade.detectMultiScale(grayImg, scaleFactor = 1.2, minNeighbors = 5)
else:
	#load lbp classifier
	lbpFaceCascade = cv2.CascadeClassifier('/home/racs/DATA/Projects/FaceDetection/xmlFiles/lbpcascade_frontalface.xml')
	faces = lbpFaceCascade.detectMultiScale(grayImg, scaleFactor = 1.2, minNeighbors = 5)


#Uses opencv to display the image in a window
#resizedImage = cv2.resize(test1, (960, 540))
#cv2.imshow('Test Img', resizedImage)
cv2.imshow('Test Img', test1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Detecting faces
#faces = haarFaceCascade.detectMultiScale(grayImg, scaleFactor = 1.2, minNeighbors = 5)

print("Faces found : ", len(faces))

#Make a rectangle over the face
for (x, y, w, h) in faces:
	cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2)

#plt.imshow(convertToRGB(test1))
cv2.imshow('Test Img', convertToRGB(test1))
cv2.waitKey(0)
cv2.destroyAllWindows()