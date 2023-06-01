import cv2 
import face_recognition

#Tespit işlemi için yüz resimlerini yükleme
messi_image = face_recognition.load_image_file("Images/messi.jpeg")
ronaldo_image = face_recognition.load_image_file("Images/ronaldo.jpeg")

#Yüklediğimiz resimleri kodlama

messi_encoding = face_recognition.face_encodings(messi_image)[0]
ronaldo_encoding = face_recognition.face_encodings(ronaldo_image)[0]

#Kodlanan yüzleri ve isimleri eşleştirme

encoding_faces = [messi_encoding,ronaldo_encoding]
name_matching = ["Messi","Ronaldo"]


#Tespit ve tanıma işleminin yapılacağı görüntüyü yükleme ve yüz tespit işlemi
img = cv2.imread("Images/messi_test.jpeg")

rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
detect_faces = face_recognition.face_locations(rgb_img)
encoding_faces_test = face_recognition.face_encodings(rgb_img,detect_faces)

find_names = []
for encoding_face_test in encoding_faces_test:
    #Bulunan yüzü tanımak için kodu karşılaştırma
    matching = face_recognition.compare_faces(encoding_faces,encoding_face_test)
    name = "Bilinmeyen"

    #Eşleşme bulunduysa ismi al
    if True in matching:
        index = matching.index(True)
        name = name_matching[index]

    find_names.append(name)

#Bulunan yüzleri ve isimlerini ekrana yazdırma
for (top,right,bottom,left), name in zip(detect_faces,find_names):
    cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)
    cv2.putText(img,name,(left,top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

#Sonucu ekranda gösterme
cv2.imshow("Face Recognition", img)
cv2.waitKey(0)
cv2.destroyAllWindows()