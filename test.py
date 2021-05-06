import cv2
import face_recognition
import os
import glob
import matplotlib.image as pltimg
import matplotlib.pyplot as plt

# Load the jpg files into numpy arrays
iu_image = face_recognition.load_image_file("IU/1.jpg")
yuna_image = face_recognition.load_image_file("yuna.jpg")
unknown_image = face_recognition.load_image_file("yuna_bv.jpg")

read_img = []
for img in glob.glob("IU/*.jpg"):
    n = cv2.imread(img)
    read_img.append(n)

try:
    iu_face_encoding = face_recognition.face_encodings(iu_image)[0]
    yuna_face_encoding = face_recognition.face_encodings(yuna_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder,filename))
#         if img is not None:
#             images.append(img)
#     return images


known_faces = [
    iu_face_encoding,
    yuna_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results = face_recognition.compare_faces(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of IU? {}".format(results[0]))
print("Is the unknown face a picture of Yuna? {}".format(results[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))
