from django.shortcuts import render

# Create your views here.
# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib # python 2
import urllib.request # python 3
import json
import cv2
import os
from sklearn.cluster import KMeans
from collections import Counter
from .imutils import resize

# define the path to the skin detector

@csrf_exempt
def detect(request):
    data = {"success": False,'skinColor':'','error':''}
    url="https://1vw4gb3u6ymm1ev2sp2nlcxf-wpengine.netdna-ssl.com/wp-content/uploads/shutterstock_149962697-946x658.jpg"
    skinColor='red'
    if request.method == "POST":
        image = _grab_image(url=url)
        image = resize(image, width=250)
        skin = extractSkin(image)
        dominantColors = extractDominantColor(skin, hasThresholding=True)
        # url = request.POST.get("url", None)
        # print(url)
        # if url is None:
        #     data.update({"skinColor": skinColor, "error":"No URL provided."})
        #     return JsonResponse(data)
        data.update({"skinColor": dominantColors,"success": True})
    return JsonResponse(data)
# def detect(request):
# 	# initialize the data dictionary to be returned by the request
# 	data = {"success": False}
# 	# check to see if this is a post request
# 	if request.method == "POST":
# 		# check to see if an image was uploaded
# 		if request.FILES.get("image", None) is not None:
# 			# grab the uploaded image
# 			image = _grab_image(stream=request.FILES["image"])
# 		# otherwise, assume that a URL was passed in
# 		else:
# 			# grab the URL from the request
# 			url = request.POST.get("url", None)
# 			# if the URL is None, then return an error
# 			if url is None:
# 				data["error"] = "No URL provided."
# 				return JsonResponse(data)
# 			# load the image and convert
# 			image = _grab_image(url=url)
# 		# convert the image to grayscale, load the face cascade detector,
# 		# and detect faces in the image
#
# 		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         skin=extractSkin(image)
# 		#update the data dictionary with the faces detected
# 		data.update({"num_faces faces success": True})
# 	# return a JSON response
# 	return JsonResponse(data)

def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)
	# otherwise, the image does not reside on disk
	else:
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.request.urlopen(url)
			data = resp.read()
		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()
		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	# return the image
	return image

def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR2YCR Colours Space to HSV
    imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)

    # Defining YCrCb Threadholds
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)

    # Single Channel mask,denoting presence of colours in the about skin region
    skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    # Return the Skin image
    return  cv2.bitwise_and(image, image, mask=skinRegionYCrCb)

def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)



    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)

def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation

def extractDominantColor(image, number_of_colors=5, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation


