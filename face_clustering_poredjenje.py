#poziv skripte python face_clustering_poredjenje.py --podaci dataset

import face_recognition
import cv2
import argparse
from os import listdir
from imutils import build_montages
from imutils import paths

#Argumenti sa komandne linije
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--podaci", required=True,help="putanja do foldera sa slikama")
ap.add_argument("-d", "--metoda", type=str, default="hog",help="metoda koja ce se koristiti `hog` ili `cnn`")
argumenti = vars(ap.parse_args())
    
#Ucitavanje slika iz odgovarajuceg foldera
putanjaDoSlika=argumenti["podaci"]
listaSlika = listdir("./"+putanjaDoSlika)
image=[]
for img in listaSlika:
    image.append(face_recognition.load_image_file(putanjaDoSlika+"/"+img))

#Kreiranje lokacija lica i kodiranje za svaku od slika
face_locations = []
face_encoding = []
konacneSlike=[]
#Putanje do slika
imagePaths = list(paths.list_images(putanjaDoSlika))
svePutanje=[]
for i in range(len(image)):
    lokacija=face_recognition.face_locations(image[i],model=argumenti["metoda"])
    #Za svaku lokaciju se dodaje novi element u nizove    
    for lok in lokacija:
        face_locations.append(lok)
        konacneSlike.append(image[i])
        svePutanje.append(imagePaths[i])
    face_encoding.append(face_recognition.face_encodings(image[i]))

image=konacneSlike
imagePaths=svePutanje

#Pretvaranje liste kodiranja u jedinstvenu listu (prvobitno je lista sastavljena od listi)
napraviJednuListu = []
for podlista in face_encoding:
    for element in podlista:
        napraviJednuListu.append(element)

face_encoding=napraviJednuListu

#Poredjenje slika
grupe=[]
indeksiSlika=[]
listaGrupa=[]
for i in range(len(face_encoding)):
    listaGrupa=[i]
    if(not(i in indeksiSlika)):
        #poznato je ono koje se nalazi na i-toj poziciji
        known_faces=[face_encoding[i]]
        for k in range(len(face_encoding)):
            if(not(k==i) and not(k in indeksiSlika)):
                ishod=face_recognition.compare_faces(known_faces, face_encoding[k])
                if(ishod[0]==True):
                    print("Da li se osoba sa %d slike nalazi i na %d. slici?" %(i+1,k+1), True)
                    listaGrupa.append(k)
                    indeksiSlika.append(k)
    if(len(listaGrupa)>1):
        grupe.append(listaGrupa)

print(grupe)

#Dodati uljeze kao krajnju grupu
uljezi=[]
napraviJednuListu = []
for podlista in grupe:
    for element in podlista:
        napraviJednuListu.append(element)
        

for i in range(len(image)):
    if(not(i in napraviJednuListu)):
        uljezi.append(i)

grupe.append(uljezi)
 
#prolazak kroz sve elemente u grupi
for i in range(len(grupe)):
    images=[]
    print("Grupe od i: ",grupe[i])
    for j in grupe[i]:
        imagePath=imagePaths[j]
        img = cv2.imread(imagePath)
        top, right, bottom, left = face_locations[j]
        face = img[top:bottom, left:right]
        face = cv2.resize(face, (96, 96))
        images.append(face)
    
    #pravljenje montaze za svaki element grupe
    montages = build_montages(images, (96, 96), (7, 7))
    for montage in montages:
        montaza=""
        if(i==(len(grupe)-1)):
            montaza="Uljezi"
        else:
            montaza="Grupa "+str(i+1)
        cv2.imshow(montaza, montage)
        cv2.waitKey(0)