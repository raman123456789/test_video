import face_recognition
from timeit import default_timer as timer
from datetime import timedelta
import time
img=['devena1.jpg','devena2.jpg','yodha.jpeg','tanveer.png','devena_yosh.jpeg','5faces.jpg']
'''
for img in img:
    print(img)

    image = face_recognition.load_image_file(img)

    start = timer()
    face_locations = face_recognition.face_locations(image, model="cnn")
    start_enc = timer()
    face_enc=face_recognition.face_encodings(image,face_locations)
    end = timer()
    print('Time to detect',timedelta(seconds=end-start))
    print ('Time to encodings.',timedelta(seconds=end-start_enc))
    time.sleep(4)
'''



image1 = face_recognition.load_image_file('devena1.jpg')
image2 = face_recognition.load_image_file('devena2.jpg')


face_locations1 = face_recognition.face_locations(image1, model="cnn")
face_locations2 = face_recognition.face_locations(image2, model="cnn")

face_enc1=face_recognition.face_encodings(image1,face_locations1)[0]
face_enc2=face_recognition.face_encodings(image2,face_locations2)[0]
start = timer()
time.sleep(4)

s=face_recognition.compare_faces([face_enc1],face_enc2)
end = timer()
print('Time to detect',timedelta(seconds=end-start))
print(s)





