from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *


params: Parameters = Parameters()

#------------------------------------------------------------------------------------------------------------------
# Introduceti aici calea (relativ la folder root=341_Dima_Cristian a folderului de test)
params.dir_test_examples = "../testare"
#------------------------------------------------------------------------------------------------------------------

#--------------------------------------------IMPORTANT-------------------------------------------------------------
# Directorul cu exemplele de test trebuie plasat in acelasi director root ca folderul 341_Dima_Cristian, astfel:

# ROOT FOLDER
# |
# | - 341_Dima_Cristian
# | 
# | - testare
#/-------------------------------------------IMPORTANT-------------------------------------------------------------

params.dir_save_files = os.path.join(params.dir_save_files, 'task2') #<- AICI SA NU UMBLATI!

params.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_hog_cell = 6  # dimensiunea celulei
params.overlap = 0.3
params.number_positive_examples = 13092  # numarul exemplelor pozitive
params.number_negative_examples = 152988 #40000  # numarul exemplelor negative # alea initiale # fara hard mining urile ulterioare

params.threshold = 0   #2.5  # toate ferestrele cu scorul > threshold si maxime locale devin detectii


facial_detector: FacialDetector = FacialDetector(params)
facial_detector.train_classifier_multiclass(None, None) # se incarca modelul deja creat din memorie

    #STAGE 1 - run
    #STAGE 2 CA FILTRARE HSV PE FETE pt a elimina FP
    
detections, scores, file_names, labels = facial_detector.run_multiclass()

low_face_hsv = (0, 30, 85) # cu sat = 0 => AP 0.639
                            # cu sat = 20 => AP 0.655
                            # cu sat = 30 => AP 0.658
high_face_hsv = (30, 100, 255)
nr_detectii_eliminate_cu_hsv_mask = 0
for i in range(len(detections)):
    xmin, ymin, xmax, ymax = detections[i]
    
    score = scores[i]
    file_name = file_names[i]
    img = cv.imread(os.path.join(facial_detector.params.dir_test_examples, file_name))
    patch = img[ymin:ymax, xmin:xmax]
    
    patch_hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV)
    mask_face_hsv = cv.inRange(patch_hsv, low_face_hsv, high_face_hsv)
    #res = cv.bitwise_and(patch_hsv, patch_hsv, mask=mask_face_hsv)
    
    # numar cat% din pixeli au ramas in res
    non_zero_count = cv.countNonZero(mask_face_hsv)
    percentage_face_pixels = non_zero_count / (patch.shape[0] * patch.shape[1])
    print(f"Percentage face pixels {file_name}, det {detections[i]}: {percentage_face_pixels}")
    
    # edge density pe fata e cam 0.2 in medie
    #try:
    # edge_density_patch = facial_detector.edge_density(patch)
    # print(f"Edge density patch {file_name}, det {detections[i]}: {edge_density_patch}")
    # except:
    #     print(f"eroare la edge density, poza {file_name}, det {detections[i]}")
    #     edge_density_patch = 1
    
    # daca mai putin de 30% din pixeli sunt in intervalul de fata, elimin detectia
    if percentage_face_pixels < 0.2: # or edge_density_patch < 0.18:  # cu 0.3 si low_fase_hsv[s_min] = 30 am AP = 0.658
                                        #cu 0.4 si low_face_hsv[s_min] = 30 am AP = 0.585
        scores[i] = -1.0  # setez un scor negativ pentru a elimina detectia
        nr_detectii_eliminate_cu_hsv_mask += 1

print("Numar initial detectii:", len(detections))
ids_ramasi = np.where(scores > facial_detector.params.threshold)[0]
final_file_names = file_names[ids_ramasi]
final_detections = detections[ids_ramasi]
final_scores = scores[ids_ramasi]
final_labels = labels[ids_ramasi]

print("Numar detectii eliminate cu HSV mask:", nr_detectii_eliminate_cu_hsv_mask)

print("Impartim detectiile pe personaje\n")

characters = ['fred', 'daphne', 'shaggy', 'velma']
        
for char in characters:
    mask = (final_labels == char)
    
    np.save(os.path.join(facial_detector.params.dir_save_files, f'detections_{char}.npy'), final_detections[mask])
    np.save(os.path.join(facial_detector.params.dir_save_files, f'scores_{char}.npy'), final_scores[mask])
    np.save(os.path.join(facial_detector.params.dir_save_files, f'file_names_{char}.npy'), final_file_names[mask])

    print(f"Salvat succes: {char} ({np.sum(mask)} detectii)")

print("\n[SUCCESS] Toate cele 12 fisiere pentru Task 2 au fost generate.")