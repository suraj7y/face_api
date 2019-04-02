import io
from _elementtree import ParseError

from django.shortcuts import render,redirect

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
known = "{base_path}/known/suraj.jpg".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))
print(known)

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
            if conf >= 4 and conf <= 45:
                # pri curl -X POST -F image=@adrian.jpg 'http://localhost:8000/face_detection/detect/' ; echo ""nt(5: #id_)
                # print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                name=labels[id_]
                color = (255, 255, 255)
                stroke = 2
                #cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        # construct a list of bounding boxes from the detection
        #rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in faces]

        # update the data dictionary with the faces detected
                data.update({"conf": conf, "Name": name, "success": True})

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

import face_recognition

@csrf_exempt
def test(request):
    if request.method == "POST":
        # check to see if an image was uploaded
        #if request.FILES.get("image", "known_img", None) is not None:
            # grab the uploaded image
        image1 = request.FILES["image"]
        print(image1)
        image = face_recognition.load_image_file(image1)
        image2 = face_recognition.load_image_file(known)
        #known_img = request.FILES["known_img"]
        face_locations = face_recognition.face_locations(image)
        known_face_encodings = face_recognition.face_encodings(image2)
        a_single_unknown_face_encoding = face_recognition.face_encodings(image)
        results = face_recognition.compare_faces(known_face_encodings, a_single_unknown_face_encoding)
        print(results)
        data={"conf": 1, "Name": "suraj", "success": True}
        return JsonResponse(data)


###################################################################################################
import face_recognition

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}




def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#@app.route('/', methods=['GET', 'POST'])
@csrf_exempt
def upload_image(request):
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return redirect(request.url)

        file = request.FILES['file']
        known = request.FILES['known']

        if file.name == '':
            return redirect(request.url)


        if file and allowed_file(file.name):
            # The image file seems valid! Detect faces and return the result.
            return detect_faces_in_image(file, known)




    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Is this a picture of Obama?</title>
    <h1>Upload a picture and see if it's a picture of Obama!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''
'''
def detect_faces_in_image1(known):
    # Pre-calculated face encoding of Obama generated with face_recognition.face_encodings(img)

    known1 = face_recognition.load_image_file(known)
    # Get face encodings for any faces in the uploaded image
    known_face_encoding = face_recognition.face_encodings(known1)

    return known_face_encoding
'''


def detect_faces_in_image(file_stream, known):
    # Pre-calculated face encoding of Obama generated with face_recognition.face_encodings(img)
    '''
    known_face_encoding = [-0.09634063,  0.12095481, -0.00436332, -0.07643753,  0.0080383,
                            0.01902981, -0.07184699, -0.09383309,  0.18518871, -0.09588896,
                            0.23951106,  0.0986533 , -0.22114635, -0.1363683 ,  0.04405268,
                            0.11574756, -0.19899382, -0.09597053, -0.11969153, -0.12277931,
                            0.03416885, -0.00267565,  0.09203379,  0.04713435, -0.12731361,
                           -0.35371891, -0.0503444 , -0.17841317, -0.00310897, -0.09844551,
                           -0.06910533, -0.00503746, -0.18466514, -0.09851682,  0.02903969,
                           -0.02174894,  0.02261871,  0.0032102 ,  0.20312519,  0.02999607,
                           -0.11646006,  0.09432904,  0.02774341,  0.22102901,  0.26725179,
                            0.06896867, -0.00490024, -0.09441824,  0.11115381, -0.22592428,
                            0.06230862,  0.16559327,  0.06232892,  0.03458837,  0.09459756,
                           -0.18777156,  0.00654241,  0.08582542, -0.13578284,  0.0150229 ,
                            0.00670836, -0.08195844, -0.04346499,  0.03347827,  0.20310158,
                            0.09987706, -0.12370517, -0.06683611,  0.12704916, -0.02160804,
                            0.00984683,  0.00766284, -0.18980607, -0.19641446, -0.22800779,
                            0.09010898,  0.39178532,  0.18818057, -0.20875394,  0.03097027,
                           -0.21300618,  0.02532415,  0.07938635,  0.01000703, -0.07719778,
                           -0.12651891, -0.04318593,  0.06219772,  0.09163868,  0.05039065,
                           -0.04922386,  0.21839413, -0.02394437,  0.06173781,  0.0292527 ,
                            0.06160797, -0.15553983, -0.02440624, -0.17509389, -0.0630486 ,
                            0.01428208, -0.03637431,  0.03971229,  0.13983178, -0.23006812,
                            0.04999552,  0.0108454 , -0.03970895,  0.02501768,  0.08157793,
                           -0.03224047, -0.04502571,  0.0556995 , -0.24374914,  0.25514284,
                            0.24795187,  0.04060191,  0.17597422,  0.07966681,  0.01920104,
                           -0.01194376, -0.02300822, -0.17204897, -0.0596558 ,  0.05307484,
                            0.07417042,  0.07126575,  0.00209804]
    '''

    # Get face encodings for any faces in the uploaded image
    known1 = face_recognition.load_image_file(known)
    # Get face encodings for any faces in the uploaded image
    known_face_encoding = face_recognition.face_encodings(known1)[0]
    #print(known_face_encoding)
    #known_face_encoding = detect_faces_in_image1()
    #known_face_encoding = known_face_encodin.known_face_encoding
    #print(known_face_encoding.known_face_encoding)
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    # Get face encodings for any faces in the uploaded image
    unknown_face_encodings = face_recognition.face_encodings(img)[0]
    #print(unknown_face_encodings)

    face_found = False
    match = False

    if len(unknown_face_encodings) > 0:
        face_found = True
        # See if the first face in the uploaded image matches the known face of Obama
        match_results = face_recognition.compare_faces([known_face_encoding], unknown_face_encodings)
        if match_results[0]:
            match = True

    # Return the result as json
    result = {
        "face_found_in_image": face_found,
        "success": match
    }
    return JsonResponse(result)