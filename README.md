# SIGN LANGUAGE TRANSLATOR

## Introduction
This repo contain a system that was made for a sign language translation model. The model was created using Grey Level Co-ocurrence Level Matrix and CNN Algorithm (GLCM-CNN) for feature extraction and classification respectively. This code is part of a Final Year Project Development Catagory (link to be added or not)

## PC Spec
Throughout the development of this repo the following system has been used:
###### Lenovo 330-15IKB
- OS : Windows 10 Home
- Processor :	Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz   1.80 GHz
- RAM :	20.0 GB (4.0 GB Built-in | 16.0 Installed)
- GPU : Intel UHD Graphixs 620
- Storage : 1 TB HDD & 500 GB SSD 

## Requirement
The following is needed to run this system:
- Python 3.9
- HTML, CSS, Javascript
- Flask (as the server*)
- Python venv (The list of installed dependencies is in requirement.txt**)

\* Only use for development server, not suitable for a production server deployment.

\** There are some that I end up not that I haven't removed so yeah...
## System Features
This system contains only 2 feature:
- Sign to Text Translator (Input: Upload)
- Fingerspelling

## How it works
The way how the translation work
1. The image is uploaded
2. It is passed to the flask server and is run through the GLCM Preprocessing which output an 1D array of the 24 GLCM value.
3. This is then passed back to the Javascript and is fed into the Model prediction method.
4. The returning value act as an index to print out what Alphabet it was matched to 

## Screenshots
*Home Page*
![Home](https://user-images.githubusercontent.com/37112149/212813451-d00af7c0-8be7-4b2a-846a-ac4b29a0e792.png)
*Navigation bar*
![Nav](https://user-images.githubusercontent.com/37112149/212813456-af6e0399-677a-4c68-b32e-09c6ecfebf25.png)
*Fingerspelling page*
![Dict](https://user-images.githubusercontent.com/37112149/212813444-891ea1b1-af9d-423d-99b6-803a80561874.png)
*Fingerspelling page(select sign)*
![Spelling](https://user-images.githubusercontent.com/37112149/212813460-b844c602-63ab-4126-9af4-9270881c79d2.png)
*Translate Page*
![Translate](https://user-images.githubusercontent.com/37112149/212813466-58c55ec9-9eea-467c-a649-3d72dfdd26d0.png)
*Translate Page(Translating the Gesture A)*
![Translator](https://user-images.githubusercontent.com/37112149/212813468-68b313eb-d77e-42fb-80ac-3f2593067ab7.png)
*About Page*
![About](https://user-images.githubusercontent.com/37112149/212813436-b9537e51-c017-4e42-af1c-ce0b7873849b.png)


## Issues
- Can only make prediction by upload because it is unable to make inference in real time due to the need of sending the video feed to the server to preprocess before passing it through the tensoflow.js model prediction method. Might be able to solve if we integrate the GLCM into a preprocessing layer.
- Due to the dataset being a video that is just chopped into the frame, there is not much of a diversity in picture / background. Thus the model is bad at making prediction with picture that has a complex or a different background. Maybe a larger dataset with more variety might help.
- The model can only detect the sign if it is the main focus of the picture. Maybe adding a regression to detect the position might solve this problem.
- The fingerspelling asset are directly in the folder, might cause problem once the rage of sign is expanded. 
- Have added a dark mode button but haven't really stylized the whole page for dark mode

## References
- The model itself (TBA)
- Tensorflow.js
