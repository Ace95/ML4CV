from hashlib import new
from operator import ne
import tensorflow as tf
import pandas as pd
import cv2
import csv

df = pd.read_csv("C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/HAM10000_metadata.csv", dtype=str)

df.set_index("dx", inplace=True, drop=False)
akiec_df = df.loc["akiec"]
bcc_df = df.loc["bcc"]
bkl_df = df.loc["bkl"]
df_df = df.loc["df"]
mel_df = df.loc["mel"]
vasc_df = df.loc["vasc"]

nameList = []

for i in range(0,len(akiec_df)):
    filename = "C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/Images/" + akiec_df.iloc[i,1]
    img = cv2.imread(filename)
    #rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    #flipped_img = cv2.flip(img, 1)
    spec_img = cv2.flip(img, 0)
    newname = "C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/SpecularB/"+ "SPECB" + "_" + akiec_df.iloc[i,1] 
    cv2.imwrite(newname,spec_img)
    nameList.append(akiec_df.iloc[i,1:7])

for i in range(0,len(bcc_df)):
    filename = "C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/Images/" + bcc_df.iloc[i,1]
    img = cv2.imread(filename)
    #rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    #flipped_img = cv2.flip(img, 1)
    spec_img = cv2.flip(img, 0)
    newname = "C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/SpecularB/"+ "SPECB" + "_" + bcc_df.iloc[i,1] 
    cv2.imwrite(newname,spec_img)
    nameList.append(bcc_df.iloc[i,1:7])

for i in range(0,len(bkl_df)):
    filename = "C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/Images/" + bkl_df.iloc[i,1]
    img = cv2.imread(filename)
    #rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    #flipped_img = cv2.flip(img, 1)
    spec_img = cv2.flip(img, 0)
    newname = "C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/SpecularB/"+ "SPECB" + "_" + bkl_df.iloc[i,1] 
    cv2.imwrite(newname,spec_img)
    nameList.append(bkl_df.iloc[i,1:7])

for i in range(0,len(df_df)):
    filename = "C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/Images/" + df_df.iloc[i,1]
    img = cv2.imread(filename)
    #rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    #flipped_img = cv2.flip(img, 1)
    spec_img = cv2.flip(img, 0)
    newname = "C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/SpecularB/"+ "SPECB" + "_" + df_df.iloc[i,1] 
    cv2.imwrite(newname,spec_img)
    nameList.append(df_df.iloc[i,1:7])

for i in range(0,len(mel_df)):
    filename = "C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/Images/" + mel_df.iloc[i,1]
    img = cv2.imread(filename)
    #rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    #flipped_img = cv2.flip(img, 1)
    spec_img = cv2.flip(img, 0)
    newname = "C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/SpecularB/"+ "SPECB" + "_" + mel_df.iloc[i,1] 
    cv2.imwrite(newname,spec_img)
    nameList.append(mel_df.iloc[i,1:7])

for i in range(0,len(vasc_df)):
    filename = "C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/Images/" + vasc_df.iloc[i,1]
    img = cv2.imread(filename)
    #rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    #flipped_img = cv2.flip(img, 1)
    spec_img = cv2.flip(img, 0)
    newname = "C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/SpecularB/"+ "SPECB" + "_" + vasc_df.iloc[i,1] 
    cv2.imwrite(newname,spec_img)
    nameList.append(vasc_df.iloc[i,1:7])

file = open('C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/SpecularB/specularimgsB.csv', 'w+', newline ='')

with file:
    write = csv.writer(file)
    write.writerows(nameList)