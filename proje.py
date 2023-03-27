import cv2
import numpy as np

# Giriş görüntüsünü yükle
image = cv2.imread("resim.jpg")

# Yüz tanıma modelini yükle
face_detection_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Griye dönüştür ve yüzleri bul
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_detection_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Kişi sayısı için sayacı tanımla
male_count = 0
female_count = 0

# Her yüz için cinsiyet tahmini yap
for (x, y, w, h) in faces:
    # Yüz bölgesini kırp
    face = image[y:y+h, x:x+w]

    # Cinsiyet tahmini yapmak için görüntüyü boyutlandır
    resized_face = cv2.resize(face, (96, 96))

    # Cinsiyet tahmini yap
    gender_classifier = cv2.dnn.readNetFromCaffe("gender.prototxt", "gender.caffemodel")
    blob = cv2.dnn.blobFromImage(resized_face, 1.0, (96, 96), (104.0, 177.0, 123.0))
    gender_classifier.setInput(blob)
    predictions = gender_classifier.forward()
    gender = "Erkek" if predictions[0][0] > predictions[0][1] else "Kadın"

    # Cinsiyete göre kişi sayısını arttır
    if gender == "Erkek":
        male_count += 1
    else:
        female_count += 1

# Sonuçları yazdır
print("Toplam Kişi Sayısı:", len(faces))
print("Erkek Sayısı:", male_count)
print("Kadın Sayısı:", female_count)
