CSCI 59000 Biometrics - face recognition
Author: Chu-An Tsai
04/09/2020

1. Dataset: Yale Face Database
    http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip

2. Platform: Matlab

3. In this assignment, I recrop the face images from the Yale face database and normalize each photo to 160x120 pixels. I use partial dataset with 9 people, and each person has 9 different faces. I select 6 different faces from each to construct a total 54-photo training dataset, and other 27 photos plus 1 from Einstein and 1 form Trump forms the test dataset. I follow the approach from Kyungnam Kim, converting the 2-D images into 1-D vetors and get the mean face vector. Then calculate the difference between each person's face vector and the mean face vector, and combine these vectors into a sample face matrix. Following is to perform PCA on the sample face matrix with a theshold on 90% and get the eigen-face matrix and also 54 projected vectors for each training face. Then I import one test image at the time and also convert the test image into a vector, and get the projected vector by multiplying the test image vector with the eigen-face vector. Finally, I calculate the Euclidean distance between the projected vector and all 54 projected vectors to find the closest three faces.

4. Reference:
  (1). Kyungnam Kim. Face Recognition using Principle Component Analysis [J] . IEEESignal Processing Society，2002，9（2）：40-42.
  (2). Turk,M.A. ; Media Lab., MIT, Cambridge, MA, USA ; Pentland, A.P. Face Recognitionusing Eigenfaces [J] . IEEE Signal Processing Society，1991，7