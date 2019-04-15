
from django.shortcuts import render,redirect

# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import os
import face_recognition

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#BASE_DIR1 = os.path.dirname(os.path.dirname(os.path.abspath('/knn/train/')))
#print(BASE_DIR1)

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

        if file.name == '':
            return redirect(request.url)


        #if file and allowed_file(file.name):
            # The image file seems valid! Detect faces and return the result.
         #   return detect_faces_in_image(file, known)

        if file and allowed_file(file.name):
            # The image file seems valid! Detect faces and return the result.
            model = os.path.join(BASE_DIR, 'trained_knn_model.clf')
            return predict(file, model_path=model)





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

@csrf_exempt
def data_trainning(request):
    if request.method == 'POST':
        if 'img1' not in request.FILES:
            return redirect(request.url)


        file = request.FILES['img1']

        img_extension = os.path.splitext(file.name)[1]

        #known = request.FILES['known']

        #print("trained")
        dirname = request.POST['id']
        path1 = 'knn/train/'+dirname+'/'
        #print(BASE_DIR1)
        file_name = os.mkdir(os.path.join(BASE_DIR, path1))
        print(file_name)
        newpath  = os.path.join(BASE_DIR, path1)
        print(newpath+"jjdjjdjjjjjjjjjjjjjjjjjjjjjjjj")

        img_save_path = newpath+ 'avatar'+img_extension


        return handle_uploaded_file(img_save_path, file)


    return '''
    <!doctype html>
    <title>Is this a picture of Obama?</title>
    <h1>Upload a picture and see if it's a picture of Obama!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

def training(request):
    try:
        tains = os.path.join(BASE_DIR, 'knn/train')
        train(tains, model_save_path="trained_knn_model.clf", n_neighbors=2)

        result = {

            "data training ": "data trained",
            "success": True
        }
        return JsonResponse(result)
    except:
        result = {

            "success": False

        }
        return JsonResponse(result)

def handle_uploaded_file(file_name,f):
    print(f)
    try:
        with open(file_name, 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
            destination.close()

            result = {
                "file": "uploded",
                "success": True

            }
        return JsonResponse(result)
    except:
        result = {

            "success": False

        }
        return JsonResponse(result)


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
    #print(unknown_face_encodings)
    if len(unknown_face_encodings) > 0:
        face_found = True
        # See if the first face in the uploaded image matches the known face of Obama
        match_results = face_recognition.compare_faces([known_face_encoding], unknown_face_encodings)
        print(match_results)
        if match_results[0]:
            match = True

    # Return the result as json
    result = {
        "face_found_in_image": face_found,
        "success": match
    }
    return JsonResponse(result)


##########################################################################################################

import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
from face_recognition.face_recognition_cli import image_files_in_folder


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):

    X = []
    y = []

    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:

                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:

                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.4):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    #if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
     #   raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
            print(knn_clf)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    predictions=[(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
     zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
    for name, (top, right, bottom, left) in predictions:
        print("- Found {} at ({}, {})".format(name, left, top))
        if name == 'unknown':
            success = False
        else:
            success =True
        result = {
            "id": name,
            "success":success

        }

    # Predict classes and remove classifications that aren't within the threshold
        return JsonResponse(result)


"""
def show_prediction_labels_on_image(img_path, predictions):

    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()

"""
"""
if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    for image_file in os.listdir("knn_examples/test"):
        full_file_path = os.path.join("knn_examples/test", image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        # Display results overlaid on an image
    # show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)
"""