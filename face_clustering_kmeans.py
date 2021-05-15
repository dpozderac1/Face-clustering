#poziv skripte python face_clustering_kmeans.py --podaci dataset --broj-klastera 10
import face_recognition
import cv2
import argparse
import numpy as np
from os import listdir
import numpy as np
from imutils import build_montages
from imutils import paths
from sklearn.cluster import KMeans

#Argumenti sa komandne linije
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--podaci", required=True,help="putanja do foldera sa slikama")
ap.add_argument("-e", "--broj-klastera", required=True,type=int, help="broj klastera koje ce koristiti kmeans algoritam(broj mora biti manji od broja slika u folderu)")
ap.add_argument("-d", "--metoda", type=str, default="hog",help="metoda koja ce se koristiti \"hog\" ili \"cnn\"")
argumenti = vars(ap.parse_args())


#Ucitavanje slika iz odgovarajuceg foldera
putanjaDoSlika=argumenti["podaci"]
listaSlika = listdir("./"+putanjaDoSlika)


#Putanje do slika
imagePaths = list(paths.list_images(putanjaDoSlika))
brojSlika=len(imagePaths)
svePutanje=[]
face_encoding = []
#Kreiranje lokacija lica za svaku od slika
face_locations = []
for i in range(len(imagePaths)):
    slika = cv2.imread(imagePaths[i])
    rgb = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
    lokacija=face_recognition.face_locations(rgb,model=str(argumenti["metoda"]))
    for lok in lokacija:
        face_locations.append(lok)
        svePutanje.append(imagePaths[i])
    encoding=face_recognition.face_encodings(rgb,lokacija)
    face_encoding.append(encoding)

imagePaths=svePutanje

napraviJednuListu = []
for podlista in face_encoding:
    for element in podlista:
        napraviJednuListu.append(element)

face_encoding=napraviJednuListu


#Clustering
#Broj klastera za kmeans
brojKlastera=argumenti["broj_klastera"]
if(brojKlastera>brojSlika):
    raise Exception("Broj klastera je veci od broja slika u skupu podataka!")
klasteri = KMeans(n_clusters=int(brojKlastera))
klasteri.fit(face_encoding)

sveLabele = np.unique(klasteri.labels_)
# loop over the unique face integers
for labela in sveLabele:
    idxs = np.where(klasteri.labels_ == labela)[0]
    licaIzGrupe = []
    for i in idxs:
        img = cv2.imread(imagePaths[i])
        #for j in range(len(face_locations[i])):
        top, right, bottom, left = face_locations[i]
        face = img[top:bottom, left:right]
        face = cv2.resize(face, (96, 96))
        licaIzGrupe.append(face)		
    #pravljenje montaze
    montaza = build_montages(licaIzGrupe, (96, 96), (7, 7))[0]
	
    nazivGrupe=""
    if(labela!=-1):
        nazivGrupe="Grupa %d"%labela
    else:
        nazivGrupe="Uljezi"
    cv2.imshow(nazivGrupe, montaza)
    cv2.waitKey(0)