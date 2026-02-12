1. the libraries required to run the project including the full version of each library.

NumPy: 2.1.3
OpenCV: 4.12.0
ultralytics: 8.4.4  # pentru inferenta cu YOLO in cadrul bonus 1 si bonus 2
scikit-learn: 1.6.1


2. how to run my code and where to look for the output files.

Structura proiectului:

Directorul 341_Dima_Cristian contine:

- fisiereSolutie [folder]
---- 341_Dima_Cristian [folder]
-------- bonus  [folder]
----------- task1 [folder]    <- aici vor fi salvate fisierele .npy pentru bonus task 1
----------- task2 [folder]    <- aici vor fi salvate fisierele .npy pentru bonus task 2
-------- task 1 [folder]      <- aici vor fi salvate fisierele .npy pentru task 1
-------- task 2 [folder]      <- aici vor fi salvate fisierele .npy pentru task 2

- modeleSVM [folder]
---- task1 [folder]           <- contine fisierul de unde e incarcat modelul SVM liniar utilizat la task1
---- task2 [folder]           <- contine fisierul de unde e incarcat modelul SVM liniar utilizat la task2

- yolo_finetunat [folder]     <- contine fisierul .pt de unde e incarcat yolo finetunat pentru rezolvarea bonusului

- FacialDetector.py [fisier]  <- contine clasa utilizata pentru rularea modelului SVM Linear/Sliding window/HOG pentru taskurile 1 si 2
- Parameters.py [fisier]      <- contine clasa parametrilor modelului de detectare faciala
- Visualize.py

- run_bonus12.py [fisier]     <- trebuie rulat pentru generarea solutiei pentru bonus task 1 si bonus task 2
- run_task1.py [fisier]       <- trebuie rulat pentru generarea solutiei la taskul 1
- run_task2.py [fisier]       <- trebuie rulat pentru generarea solutiei la taskul 2


Pentru a rula proiectul:

1.                                                PENTRU TASK 1

- mergeti in run_task1.py
- pe linia 11:

 9  #------------------------------------------------------------------------------------------------------------------
10  # Introduceti aici calea (relativ la folder root=341_Dima_Cristian a folderului de test)
11  params.dir_test_examples = "../validare/validare"
12  #------------------------------------------------------------------------------------------------------------------

- inlocuiti cu folderul care contine exemplele de test
- acest folder cu fisierele de test ar trebui plasat in acelasi director root ca folderul meu mare 341_Dima_Cristian, astfel:

# ROOT FOLDER
# |
# | - 341_Dima_Cristian
# | 
# | - testare

- dupa inlocuirea path-ului, rulati
- cele 3 fisiere .npy cerute de taskul 1 vor aparea in fisiereSolutie/341_Dima_Cristian/task1



2.                                                PENTRU TASK 2

- mergeti in run_task2.py
- pe linia 11:

 9  #------------------------------------------------------------------------------------------------------------------
10  # Introduceti aici calea (relativ la folder root=341_Dima_Cristian a folderului de test)
11  params.dir_test_examples = "../validare/validare"
12  #------------------------------------------------------------------------------------------------------------------

- inlocuiti cu folderul care contine exemplele de test
- acest folder cu fisierele de test ar trebui plasat in acelasi director root ca folderul meu mare 341_Dima_Cristian, astfel:

# ROOT FOLDER
# |
# | - 341_Dima_Cristian
# | 
# | - testare

- dupa inlocuirea path-ului, rulati
- cele 12 fisiere .npy cerute de taskul 3 vor aparea in fisiereSolutie/341_Dima_Cristian/run_task2


3.                                    PENTRU BONUS TASK 1 SI BONUS TASK 2

- mergeti in run_bonus12.py
- pe linia 8:

 5   #-------------------------------------------------------------------------------------------------------------------------
 6   # INLOCUITI CU CALEA (RELATIVA LA FOLDERUL ROOT 341_Dima_Cristian) CATRE FOLDERUL CU IMAGINILE DE TEST
 7
 8   test_images_path = "../validare/validare"
 9
10   #-------------------------------------------------------------------------------------------------------------------------

- inlocuiti cu folderul care contine exemplele de test
- acest folder cu fisierele de test ar trebui plasat in acelasi director root ca folderul meu mare 341_Dima_Cristian, astfel:

# ROOT FOLDER
# |
# | - 341_Dima_Cristian
# | 
# | - testare

- dupa inlocuirea path-ului, rulati
- cele 3 fisiere .npy cerute de bonus task 1 vor aparea in fisiereSolutie/341_Dima_Cristian/bonus/task1
- cele 12 fisiere .npy cerute de bonus task 2 vor aparea in fisiereSolutie/341_Dima_Cristian/bonus/task2
