import face_recognition
image = face_recognition.load_image_file("IU/1.jpg")
face_locations = face_recognition.face_locations(image)


known_image = image #face_recognition.load_image_file("biden.jpg")
unknown_image = face_recognition.load_image_file("joe.png")

known_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([known_encoding], unknown_encoding)


if results[0] == True:
    print('It\'s IU!')
else:
    print('Who are you?')
