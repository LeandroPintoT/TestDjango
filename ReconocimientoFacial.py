import numpy as np
import cv2
import os


subjects = ["Desconocido", "Leandro Pinto", "Robert Downey Jr", "Emma Watson", "Brad Pitt", "Scarlett Johansson", "Marcos Rojas", "Nicolas Mura"]


def detect_face(img):
    # convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow: Haar classifier
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

    # let's detect multiscale images(some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)

    # if no faces are detected then return original img
    if len(faces) == 0:
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area
    grays = []
    for face in faces:
        (x, y, w, h) = face
        grays.append(gray[y:y + w, x:x + h])

    # return only the face part of the image
    return grays, faces


# this function will read all persons' training images, detect face from each image
# and will return two lists of exactly same size, one list
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path=os.getcwd()):
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("f"):
            continue

        # ------STEP-2--------
        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("f", ""))

        # build path of directory containing images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)

            # detect face
            facees, rect = detect_face(image)
            if facees is not None:
                for face in facees:
                    # ------STEP-4--------
                    # for the purpose of this tutorial
                    # we will ignore faces that are not detected
                    if face is not None:
                        # add face to list of faces
                        print face
                        faces.append(face)
                        # add label for this face
                        labels.append(label)
                    else:
                        print image_path

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data()
print("Data prepared")

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer2 = cv2.face.LBPHFaceRecognizer_create()

# or use EigenFaceRecognizer by replacing above line with
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

# or use FisherFaceRecognizer by replacing above line with
#face_recognizer = cv2.face.FisherFaceRecognizer_create()

# train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
    # make a copy of the image as we don't want to change original image
    img = test_img.copy()
    # detect face from the image
    faces, rect = detect_face(img)

    i = 0
    try:
        for face in faces:
            # predict the image using our face recognizer
            label = face_recognizer.predict(face)

            # get name of respective label returned by face recognizer
            print label

            label_text = subjects[label[0]]

            # draw a rectangle around face detected
            draw_rectangle(img, rect[i])
            # draw name of predicted person
            draw_text(img, label_text, rect[i][0], rect[i][1] - 5)
            i = i + 1

    except Exception:
        pass
    return img


cap = cv2.VideoCapture(1)
while True:
    ret, test_img = cap.read()
    predicted_img = predict(test_img)
    cv2.imshow('Zukarita', predicted_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
