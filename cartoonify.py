import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "Path to the image",default='./test_images/dicaprio.jpg')
args = vars(ap.parse_args())

def mask_creation(img):
  #The function generates a edge mask using adaptive thresholding
  gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  gray_blur = cv2.medianBlur(gray_image, 5)
  mask_ = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 6)
  return mask_


def color_mapping(img):
  #This function actually limits the colours combinations which is needed to be appeared in the cartoon canvas
  data = np.float32(img).reshape((-1, 3))
  #setting up the mapping criteria
  mapping_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.25)
  #doing k means for color quantization
  ret, label, center = cv2.kmeans(data, 9, None, mapping_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result
  
def cartoonify(image):
    #This is a helper function for post processing
    mask_created = mask_creation(image)

    img1 = color_mapping(image)

    #reducing the noise by usib bilateral filter
    image_test = cv2.bilateralFilter(img1, 8, 200,200)
    #adding both image and mask together
    cartoon = cv2.bitwise_and(image_test, image_test, mask=mask_created)
    cv2.imwrite("cartoonized.jpg",cartoon)
    

    
img = cv2.imread(args["image"])
cartoonify(img)
print("Cartoon Creation is Completed")
    
    

    