import io
from _elementtree import ParseError

from django.shortcuts import render

# Create your views here.
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer, BrowsableAPIRenderer
from rest_framework import status
from rest_framework.response import Response
import json
from rest_framework.parsers import JSONParser,FormParser, MultiPartParser, FileUploadParser
from rest_framework.decorators import parser_classes

# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
import os
import pickle

# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

face_cascade1 = "{base_path}/cascades/data/haarcascade_frontalface_alt2.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
#recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.read("{base_path}/recognizers/face-trainner.yml".format(
 #   base_path=os.path.abspath(os.path.dirname(__file__))))

@csrf_exempt
def detect(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)

        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # construct a list of bounding boxes from the detection
        rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

        # update the data dictionary with the faces detected
        data.update({"num_faces": len(rects), "faces": rects, "success": True})

    # return a JSON response
    return JsonResponse(data)


@csrf_exempt
def detect1(request):
    #recognizer = cv2.face.LBPHFaceRecognizer_create()
    #recognizer.read("./recognizers/face-trainner.yml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("{base_path}/recognizers/face-trainner.yml".format(
        base_path=os.path.abspath(os.path.dirname(__file__))))
    with open("{base_path}/pickles/face-labels.pickle".format(
    base_path=os.path.abspath(os.path.dirname(__file__))), 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)

        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(face_cascade1)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          )
        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
            #roi_color = frame[y:y + h, x:x + w]

            # recognize? deep learned model predict keras tensorflow pytorch scikit learn
            id_, conf = recognizer.predict(roi_gray)
            rects=[]
            #name=[]
            if conf >= 4 and conf <= 55:
                # print(5: #id_)
                # print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name=labels[id_]
                color = (255, 255, 255)
                stroke = 2
                #cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        # construct a list of bounding boxes from the detection
        #rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in faces]

        # update the data dictionary with the faces detected
                data.update({"num_faces": len(name), "faces": name, "success": True})

    # return a JSON response
    return JsonResponse(data)


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