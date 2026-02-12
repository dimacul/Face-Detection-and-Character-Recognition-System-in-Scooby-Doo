from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog
from datetime import datetime

# pt multiclass plot confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class FacialDetector:
    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None
        
    def edge_density(self, patch_bgr,
                low_thr=50,
                high_thr=150):
        """
        Returnează edge density ∈ [0,1]
        """
        gray = cv.cvtColor(patch_bgr, cv.COLOR_BGR2GRAY)

        edges = cv.Canny(gray, low_thr, high_thr)

        num_edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]

        return num_edge_pixels / total_pixels
        
    def gamma_normalize_gray(self, img_uint8, gamma=0.5):
        img = img_uint8.astype(np.float32) / 255.0
        img = np.power(img, gamma)
        return (img * 255.0).astype(np.uint8)

        
    def extract_annotations_from_char_name_train_folder(self, char_name): 
        with open(os.path.join(self.params.base_dir, f'../../antrenare/{char_name}_annotations.txt')) as f:
            #rf"C:\Users\crist\Documents\FACULTATE\ANUL 3\CAVA\LAB\TEMA2\antrenare\{char_name}_annotations.txt") as f:
            lines = f.readlines()

        file_names = []
        detections = []
        #character_names = []

        for line in lines:
            fn, xmin, ymin, xmax, ymax, char_name = line.split()
            bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
            file_names.append(fn)
            detections.append(bbox)
            #character_names.append(char_name)
            
        # return {
        #     "file_names": file_names,
        #     "detections": detections,
        #     "character_names": character_names
        # }
        
        # gt_bboxes_dict = {}
        # for i in range(len(detections)):
        #     gt_bboxes_dict[file_names[i]] = gt_bboxes_dict.get(file_names[i], []).append(detections[i])
        
        # return gt_bboxes_dict
        
        gt_bboxes_dict = {}
        for fn, bbox in zip(file_names, detections):
            gt_bboxes_dict.setdefault(fn, []).append(bbox)
        return gt_bboxes_dict


    def get_positive_descriptors(self):
        # in aceasta functie calculam descriptorii pozitivi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor pozitive
        # iar D - dimensiunea descriptorului
        # D = (params.dim_window/params.dim_hog_cell - 1) ^ 2 * params.dim_descriptor_cell (fetele sunt patrate)

        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            # TODO: sterge
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)
            print(len(features))

            positive_descriptors.append(features)
            # if self.params.use_flip_images:
            #     features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
            #                    cells_per_block=(2, 2), feature_vector=True)
            #     positive_descriptors.append(features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def get_negative_descriptors(self):
        # in aceasta functie calculam descriptorii negativi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor negative
        # iar D - dimensiunea descriptorului
        # avem 274 de imagini negative, vream sa avem self.params.number_negative_examples (setat implicit cu 10000)
        # de exemple negative, din fiecare imagine vom genera aleator self.params.number_negative_examples // 274
        # patch-uri de dimensiune 36x36 pe care le vom considera exemple negative

        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        num_negative_per_image = self.params.number_negative_examples // num_images
        negative_descriptors = []
        print('Calculam descriptorii pt %d imagini negative' % num_images)
        for i in range(num_images):
            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            # TODO: completati codul functiei in continuare
            # num_rows = img.shape[0]
            # num_cols = img.shape[1]
            # x = np.random.randint(low=0, high=num_cols - self.params.dim_window, size=num_negative_per_image)
            # y = np.random.randint(low=0, high=num_rows - self.params.dim_window, size=num_negative_per_image)

            # for idx in range(len(y)):
            #     patch = img[y[idx]: y[idx] + self.params.dim_window, x[idx]: x[idx] + self.params.dim_window]
            #     descr = hog(patch, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
            #                 cells_per_block=(2, 2), feature_vector=False)
            #     negative_descriptors.append(descr.flatten())
            descr = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                            cells_per_block=(2, 2), feature_vector=False)
            negative_descriptors.append(descr.flatten())

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors
    
    def get_positive_descriptors_val(self):
        positive_patches_val_path = os.path.join(self.params.base_dir, 'exemplePozitiveValidare')        
        val_positive_descriptors = []
        
        # calculez descriptorii pentru exemplele pozitive de validare
        print("Calculam descriptorii pentru exemplele pozitive de validare...")
        pos_files = glob.glob(os.path.join(positive_patches_val_path, '*.jpg'))
        for i in range(len(pos_files)):
            img = cv.imread(pos_files[i], cv.IMREAD_GRAYSCALE)
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)
            val_positive_descriptors.append(features)
            
        return np.array(val_positive_descriptors)
    
    def get_negative_descriptors_val(self):
        negative_patches_val_path = os.path.join(self.params.base_dir, 'exempleNegativeValidare')        
        val_negative_descriptors = []
        
        # calculez descriptorii pentru exemplele negative de validare
        print("Calculam descriptorii pentru exemplele negative de validare...")
        neg_files = glob.glob(os.path.join(negative_patches_val_path, '*.jpg'))
        for i in range(len(neg_files)):
            img = cv.imread(neg_files[i], cv.IMREAD_GRAYSCALE)
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)
            val_negative_descriptors.append(features)
            
        return np.array(val_negative_descriptors)

    def train_classifier(self, training_examples, train_labels, use_validation_set_for_performance = False,val_examples=[], val_labels=[], stage2=False):
        if not stage2:
            svm_file_name = os.path.join(self.params.dir_load_model_task1, 'best_model_%d_%d_%d' % #_16ian
                                        (self.params.dim_hog_cell, self.params.number_negative_examples,
                                        self.params.number_positive_examples))
        else:
            svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_stage2_%d_%d_%d' %
                                        (self.params.dim_hog_cell, self.params.number_negative_examples,
                                        self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            print(f"Model incarcat: {svm_file_name}")
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]#, 10, 100]
        
        # C mare -> regularizare mica -> overfitting
            # penalizează foarte tare:
            # - pozitive clasificate ca negative
            # - negative clasificate ca pozitive

            # încearcă să separe chiar și exemplele foarte dificile
            # produce:
                # margine mică
                # |w| mare
                # scoruri comprimate / instabile

            # - Favorizează exemplele HARD (mai ales hard negatives)
            
        # C mic -> regularizare mare -> underfitting
            # permite:
            # - unele greșIeli pe train
            # - ignorarea outlierilor
            # produce:
                # margine mare
                # |w| mai mic
                # scoruri mai stabile

            # - Favorizeaza structura generală (pozitivele curate)
        # if stage2:
        #     Cs = [0.5]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            if not stage2:
                model = LinearSVC(C=c, class_weight="balanced")
            else:
                model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            
            #acc pe train
            acc = model.score(training_examples, train_labels)
            print(f"Acc pe train: {acc}")
            
            if use_validation_set_for_performance:
                acc_test = model.score(val_examples, val_labels)
                print(f"Acc pe val: {acc_test}")
                
            if not use_validation_set_for_performance:
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_c = c
                    best_model = deepcopy(model)
            else:
                # best model pe setul de val:
                if acc_test > best_accuracy:
                    best_accuracy = acc_test
                    best_c = c
                    best_model = deepcopy(model)
                
                
        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]

        plt.figure(figsize=(12, 5))
        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.savefig(os.path.join(self.params.dir_save_files, 'distributia_scorurilor_clasificatorului.png'))
        plt.show()
        
    def train_classifier_multiclass(self, training_examples, train_labels, use_validation_set_for_performance = False,val_examples=[], val_labels=[]):
   
        svm_file_name = os.path.join(self.params.dir_load_model_task2, 'best_model_task2_%d_%d_%d' %
                                        (self.params.dim_hog_cell, self.params.number_negative_examples,
                                        self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            print(f"Model multiclasa task 2 incarcat: {svm_file_name}")
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]#, 10, 100]
        
        # C mare -> regularizare mica -> overfitting
            # penalizează foarte tare:
            # - pozitive clasificate ca negative
            # - negative clasificate ca pozitive

            # încearcă să separe chiar și exemplele foarte dificile
            # produce:
                # margine mică
                # |w| mare
                # scoruri comprimate / instabile

            # - Favorizează exemplele HARD (mai ales hard negatives)
            
        # C mic -> regularizare mare -> underfitting
            # permite:
            # - unele greșIeli pe train
            # - ignorarea outlierilor
            # produce:
                # margine mare
                # |w| mai mic
                # scoruri mai stabile

            # - Favorizeaza structura generală (pozitivele curate)
        # if stage2:
        #     Cs = [0.5]
        for c in Cs:
            print('Antrenam un clasificator multiclass pentru c=%f' % c)
            
            model = LinearSVC(C=c, max_iter=2000) # ca sa ma asigur ca converge
            model.fit(training_examples, train_labels)
            
            #acc pe train
            acc = model.score(training_examples, train_labels)
            print(f"Acc pe train: {acc}")
            
            if use_validation_set_for_performance:
                acc_test = model.score(val_examples, val_labels)
                print(f"Acc pe val: {acc_test}")
                
            if not use_validation_set_for_performance:
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_c = c
                    best_model = deepcopy(model)
            else:
                # best model pe setul de val:
                if acc_test > best_accuracy:
                    best_accuracy = acc_test
                    best_c = c
                    best_model = deepcopy(model)
                
                
        print('Performanta clasificatorului optim multiclasa pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        #scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        # confuzia pe train:
        self.plot_confusion(best_model, training_examples, train_labels, val=False)
        if use_validation_set_for_performance:
            self.plot_confusion(best_model, val_examples, val_labels, val=True)
        
    def plot_confusion(self, model, x, y, val=False):
        
        y_pred = model.predict(x)
        cm = confusion_matrix(y, y_pred)
        
        if not val:
            title_suffix = "setul de antrenare"
        else:
            title_suffix = "setul de validare"
        print(f"\nRaport pentru {title_suffix}:\n")
        print(classification_report(y, y_pred, target_names=['Fred', 'Daphne', 'Shaggy', 'Velma', 'unknown', 'Non-Face']))     
        
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Fred', 'Daphne', 'Shaggy', 'Velma', 'unknown', 'Non-Face'],
                    yticklabels=['Fred', 'Daphne', 'Shaggy', 'Velma', 'unknown', 'Non-Face'])
        if not val:
            plt.title('Confusion Matrix - Training Set')
        else: # validare
            plt.title('Confusion Matrix - Validation Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if not val:
            plt.savefig(os.path.join(self.params.dir_save_files, 'confusion_multiclass_train.png'))
        else: #validare
            plt.savefig(os.path.join(self.params.dir_save_files, 'confusion_multiclass_val.png'))
        plt.show()
        

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size, descriptors_from_image):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        # if descriptors_from_image is not None:
        sorted_image_descriptors = descriptors_from_image[sorted_indices]
        # sorted_image_patches = patches_from_image[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        # if descriptors_from_image is not None:
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal], sorted_image_descriptors[is_maximal] #, sorted_image_patches[is_maximal]
        #return sorted_image_detections[is_maximal], sorted_scores[is_maximal]
    
    
    def run(self, return_hard_negatives=False, 
            gt_bboxes_dict=None, 
            hard_score_thresh=0.0, 
            hard_iou_thresh=0.0, 
            max_hard_per_image=10, 
            stage2 = False):
        """
        Multi-scale sliding window + NMS.

        Optional: Hard negative mining (pe TRAIN):
        - setezi return_hard_negatives=True
        - dai gt_bboxes_dict: { 'imgname.jpg': [[x1,y1,x2,y2], ...], ... }
        - se colecteaza ferestre cu score > hard_score_thresh
            si max IoU cu orice GT <= hard_iou_thresh
        - se pastreaza top max_hard_per_image per imagine (dupa scor)
        """

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)

        detections = None
        descriptors = None # ii returnez daca stage2 = True
        scores = np.array([])
        file_names = np.array([])

        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]

        num_test_images = len(test_files)

        # scale-uri (piramida)
        s = 1.2
        #scale_list = [1.0, 1.0/s, 1.0/(s*s), 1.0/(s*s*s), 1.0/(s*s*s*s), 1.0/(s*s*s*s*s), 1.0/(s*s*s*s*s*s), s, s*s]
        #scale_list = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.75, 0.6, 0.65, 0.5, 0.55, 0.4, 0.45, 0.3, 0.35, 0.2, 0.25, 0.15, s, s*s] #- cu asta obtinusem AP max
        #scale_list = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.75, 0.6, 0.65, 0.5, 0.55, 0.4, s, s*s]
        #scale_list = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.75, 0.6, 0.65, 0.5, 0.55, 0.4, 0.45, 0.3, 0.35, 0.2, 0.25, 0.15, 1.05, 1.10, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4] # incercare noua
        scale_list = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.75, 0.6, 0.65, 0.5, 0.55, 0.4, 0.45, 0.3, 0.35, 0.2, 0.25, 0.15]#, s, s*s] #- cu asta obtinusem AP max


        # hard negatives globale (descriptori + scoruri)
        hard_descriptors = []
        hard_scores = []

        # max IoU cu lista de GT
        def max_iou_with_list(bbox, gt_list):
            if gt_list is None or len(gt_list) == 0:
                return 0.0
            return max(self.intersection_over_union(bbox, gt) for gt in gt_list)

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))

            img0 = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            
            #normalizare gamma
            #img0 = self.gamma_normalize_gray(img0, gamma=0.5) #AP mai prost
            
            if img0 is None:
                continue
            H0, W0 = img0.shape[:2]

            short_name = ntpath.basename(test_files[i])

            # GT bboxes pentru imagine (doar la hard mining)
            gt_bboxes = None
            if gt_bboxes_dict is not None and short_name in gt_bboxes_dict:
                gt_bboxes = gt_bboxes_dict[short_name]

            image_scores = []
            image_detections = []
            #image_patches = []
            image_descriptors = []

            # hard negatives pe imaginea curenta
            hard_desc_img = []
            hard_scores_img = []
            hard_neg_patches_img = []

            num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - 1  # 36/6 -1 = 5

            for scale in scale_list:
                if scale == 1.0:
                    img = img0
                else:
                    new_W = int(round(W0 * scale))
                    new_H = int(round(H0 * scale))

                    # daca imaginea scalata e prea mica pentru 36x36, skip, //desi nu am in scale ceva <0.2
                    if new_W < self.params.dim_window or new_H < self.params.dim_window:
                        continue

                    interp = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
                    img = cv.resize(img0, (new_W, new_H), interpolation=interp)

                hog_descriptors = hog(
                    img,
                    pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                    cells_per_block=(2, 2),
                    feature_vector=False
                )

                num_cols = img.shape[1] // self.params.dim_hog_cell - 1
                num_rows = img.shape[0] // self.params.dim_hog_cell - 1

                # if num_cols <= num_cell_in_template or num_rows <= num_cell_in_template:
                #     continue

                # sliding in spatiul HOG
                for y in range(0, num_rows - num_cell_in_template + 1):
                    for x in range(0, num_cols - num_cell_in_template + 1):
                        descr = hog_descriptors[y:y + num_cell_in_template, x:x + num_cell_in_template].flatten()
                        score = float(np.dot(descr, w)[0] + bias)

                        # bbox in coordonate imagine scalata
                        x_min_s = int(x * self.params.dim_hog_cell)
                        y_min_s = int(y * self.params.dim_hog_cell)
                        x_max_s = x_min_s + self.params.dim_window
                        y_max_s = y_min_s + self.params.dim_window

                        # proiectare bbox la scala 1
                        if scale != 1.0:
                            inv = 1.0 / scale
                            x_min = int(round(x_min_s * inv))
                            y_min = int(round(y_min_s * inv))
                            x_max = int(round(x_max_s * inv))
                            y_max = int(round(y_max_s * inv))
                        else:
                            x_min, y_min, x_max, y_max = x_min_s, y_min_s, x_max_s, y_max_s

                        # clamp in imaginea originala
                        x_min = max(0, min(x_min, W0 - 1))
                        y_min = max(0, min(y_min, H0 - 1))
                        x_max = max(0, min(x_max, W0))
                        y_max = max(0, min(y_max, H0))

                        bbox = [x_min, y_min, x_max, y_max] # in coord imagine originala
                        
                        
                        if score > self.params.threshold:
                            image_detections.append(bbox)
                            image_scores.append(score)
                            image_descriptors.append(descr)
                            #image_patches.append(img0[y_min:y_max, x_min:x_max])

                        # hard negative mining - trebuie facut dupa NMS ca sa nu apara multe imagini aproape duplicat hard neg
                        # if return_hard_negatives and gt_bboxes_dict is not None:
                        #     if score > hard_score_thresh:
                        #         if max_iou_with_list(bbox, gt_bboxes) <= hard_iou_thresh:
                        #             hard_desc_img.append(descr.astype(np.float32))
                        #             hard_scores_img.append(score)
                        #             hard_neg_patches_img.append(img0[y_min:y_max, x_min:x_max])

            # NMS pe imagine (pe toate scale-urile, in coordonate scale=1)
            if len(image_scores) > 0:
                image_detections, image_scores, image_descriptors = self.non_maximal_suppression(
                    np.array(image_detections),
                    np.array(image_scores),
                    img0.shape,
                    np.array(image_descriptors)
                    # np.array(image_patches)
                )
                
                # hard negative mining
                if return_hard_negatives and gt_bboxes_dict is not None:
                    for score, bbox, descr in zip(image_scores, image_detections, image_descriptors):
                        if score > hard_score_thresh:
                            if max_iou_with_list(bbox, gt_bboxes) <= hard_iou_thresh:
                                x_min, y_min, x_max, y_max = bbox
                                hard_desc_img.append(descr.astype(np.float32))
                                hard_scores_img.append(score)
                                hard_neg_patches_img.append(img0[y_min:y_max, x_min:x_max])

            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                    if stage2:
                        descriptors = image_descriptors
                else:
                    detections = np.concatenate((detections, image_detections))
                    if stage2:
                        descriptors = np.concatenate((descriptors, image_descriptors))

                scores = np.append(scores, image_scores)

                image_names = [short_name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

            # pastrez max hard img hard negatives per imagine (ord dupa scor descr)
            final_hard_neg_patches_img = []
            if return_hard_negatives and len(hard_scores_img) > 0:
                idx = np.argsort(hard_scores_img)[::-1]
                idx = idx[:min(max_hard_per_image, len(idx))]   #BA DA! nu mai vreau sa pun numar max de hard examples per image
                for k in idx:
                    hard_descriptors.append(hard_desc_img[k])
                    hard_scores.append(hard_scores_img[k])
                    final_hard_neg_patches_img.append(hard_neg_patches_img[k])
                    
            if return_hard_negatives:
                print(f"Pentru imaginea: {short_name} s-au gasit {len(final_hard_neg_patches_img)} hard negative patches.")
                    
            #scriu in fisier patchurile hard neg gasite in imgaginea curenta
            if return_hard_negatives and len(final_hard_neg_patches_img) > 0:
                save_dir = os.path.join(self.params.base_dir, 'exemplePuternicNegative')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                for idx_patch, patch in enumerate(final_hard_neg_patches_img):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    patch_filename = f"hardneg_img_{short_name[:-4]}_patch_{idx_patch}_{timestamp}.jpg"
                    cv.imwrite(os.path.join(save_dir, patch_filename), patch)

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                % (i, num_test_images, end_time - start_time))

        if return_hard_negatives:
            if len(hard_descriptors) > 0:
                hard_descriptors = np.asarray(hard_descriptors, dtype=np.float32)
                hard_scores = np.asarray(hard_scores, dtype=np.float32)
            else:
                hard_descriptors = np.zeros((0, w.shape[0]), dtype=np.float32)
                hard_scores = np.zeros((0,), dtype=np.float32)

            return detections, scores, file_names, hard_descriptors, hard_scores

        if stage2:
            return detections, scores, file_names, descriptors
        return detections, scores, file_names
    
    def run_multiclass(self):
        """
        Multi-scale sliding window optimizat pentru Multiclass (Task 2).
        Folosește vectorizarea pentru viteză maximă pe CPU.
        """
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = sorted(glob.glob(test_images_path))
        num_test_images = len(test_files)

        # rezultate finale pentru fisierele .npy
        all_detections = []
        all_scores = []
        all_file_names = []
        all_labels = []
        
        mapping = {0: 'fred', 1: 'daphne', 2: 'shaggy', 3: 'velma'} # 4 = unknown, 5 = non-face

        scale_list = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            short_name = ntpath.basename(test_files[i])
            print(f'Procesam imaginea {i+1}/{num_test_images}: {short_name}')

            img0 = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            if img0 is None: 
                continue
            H0, W0 = img0.shape

            #liste pentru colectarea ferestrelor din TOATE scale-urile unei imagini
            raw_descriptors = []
            raw_bboxes = []

            #sliding window si colectez descriptorii
            for scale in scale_list:
                new_W, new_H = int(round(W0 * scale)), int(round(H0 * scale))
                if new_W < self.params.dim_window or new_H < self.params.dim_window:
                    continue

                img = cv.resize(img0, (new_W, new_H), interpolation=cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR)
                
                # Calcul HOG pe toată imaginea scalată
                hog_descriptors = hog(img, 
                                      pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                      cells_per_block=(2, 2), 
                                      feature_vector=False)

                num_cols = hog_descriptors.shape[1]
                num_rows = hog_descriptors.shape[0]
                num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - 1

                for y in range(num_rows - num_cell_in_template + 1):
                    for x in range(num_cols - num_cell_in_template + 1):
                        #descriptorul ferestrei
                        descr = hog_descriptors[y:y + num_cell_in_template, x:x + num_cell_in_template].flatten()
                        
                        #proiectez bboxul la scara 1
                        inv = 1.0 / scale
                        x_min = int(round(x * self.params.dim_hog_cell * inv))
                        y_min = int(round(y * self.params.dim_hog_cell * inv))
                        x_max = int(round((x * self.params.dim_hog_cell + self.params.dim_window) * inv))
                        y_max = int(round((y * self.params.dim_hog_cell + self.params.dim_window) * inv))

                        raw_descriptors.append(descr)
                        raw_bboxes.append([x_min, y_min, x_max, y_max])

            #Pt eficienta, in etapa asta abia: clasificare vectorizata (multiclass)
            if len(raw_descriptors) > 0:
                raw_descriptors = np.array(raw_descriptors)
                
                #calculez toate scorurile printr o sg trecere prin svm
                scores_matrix = self.best_model.decision_function(raw_descriptors)     #o matrice (N_ferestre, 6_clase)

                
                # pt fiecare fereastra identific clasa cu scor maxim + acel scor maxim
                predictions = np.argmax(scores_matrix, axis=1)
                max_confidences = np.max(scores_matrix, axis=1)

                # pastres doar personajele de interes (clase 0 -3) cu scor > threshold
                valid_mask = (predictions <= 3) & (max_confidences > self.params.threshold)
                
                img_detections = np.array(raw_bboxes)[valid_mask]
                img_scores = max_confidences[valid_mask]
                img_desc_to_nms = raw_descriptors[valid_mask] # pt NMS vreau si descriptorii

                #NMS
                if len(img_scores) > 0:
                    #nms ret detectii, scoruri, descriptori
                    final_img_det, final_img_sc, final_img_desc = self.non_maximal_suppression(img_detections, img_scores, img0.shape, img_desc_to_nms)

                    all_detections.extend(final_img_det)
                    all_scores.extend(final_img_sc)
                    all_file_names.extend([short_name] * len(final_img_sc))
                    
                    current_labels_ids = self.best_model.predict(final_img_desc)
                    for label_id in current_labels_ids:
                        all_labels.append(mapping[label_id])

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                % (i, num_test_images, end_time - start_time))


        # np.save('detections_character.npy', np.array(all_detections, dtype=object))
        # np.save('scores_character.npy', np.array(all_scores, dtype=object))
        # np.save('file_names_character.npy', np.array(all_file_names, dtype=object))

        # print("\n[SUCCESS] Fișierele pentru Task 2 au fost salvate.")
        # return np.array(all_detections), np.array(all_scores), np.array(all_file_names)
        all_detections = np.array(all_detections)
        all_scores = np.array(all_scores)
        all_file_names = np.array(all_file_names)
        all_labels = np.array(all_labels)

        
        return all_detections, all_scores, all_file_names, all_labels
    
    def run_stage2(self, prev_detections, prev_scores, prev_file_names, prev_descriptors, facial_detector_stage2):
        #dintre descriptorii returnati de run, aleg doar pe cei pentru care al doilea model
        # (cel antrenat pe pozitive + hard neg scoase de primul model) da scor peste prag
        print("Stage 2: Filtram detectiile initiale cu al doilea model pentru a elimina FP...")
        final_detections = []
        final_scores = []
        final_file_names = []
        num_detections = len(prev_detections)
        w = facial_detector_stage2.best_model.coef_.T
        bias = facial_detector_stage2.best_model.intercept_[0]

        for detection_idx in range(num_detections):
            score = float(np.dot(prev_descriptors[detection_idx], w)[0] + bias)
            if score > facial_detector_stage2.params.threshold:
                final_detections.append(prev_detections[detection_idx])
                final_scores.append(score)
                final_file_names.append(prev_file_names[detection_idx])

        return np.array(final_detections), np.array(final_scores), np.array(final_file_names)

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        with open(r"C:\Users\crist\Documents\FACULTATE\ANUL 3\CAVA\LAB\TEMA2\341_Dima_Cristian_sursa\data\salveazaFisiere\raport_validare.txt", "w") as f:
            ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
            ground_truth_file_names = np.array(ground_truth_file[:, 0])
            ground_truth_detections = np.array(ground_truth_file[:, 1:], dtype=int) #np.int nu merge

            num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
            gt_exists_detection = np.zeros(num_gt_detections)
            # sorteazam detectiile dupa scorul lor
            sorted_indices = np.argsort(scores)[::-1]
            file_names = file_names[sorted_indices]
            scores = scores[sorted_indices]
            detections = detections[sorted_indices]

            num_detections = len(detections)
            true_positive = np.zeros(num_detections)
            false_positive = np.zeros(num_detections)
            duplicated_detections = np.zeros(num_detections)

            for detection_idx in range(num_detections):
                indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

                gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
                bbox = detections[detection_idx]
                max_overlap = -1
                
                best_gt_local = -1
                
                
                index_max_overlap_bbox = -1
                for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                    overlap = self.intersection_over_union(bbox, gt_bbox)
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_gt_local = gt_idx
                        index_max_overlap_bbox = indices_detections_on_image[gt_idx] # indexul global  care apartine detectiilor pe imag curenta   si are max overlap

                f.write(f"\nImagine: {file_names[detection_idx]}")
                # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
                if max_overlap >= 0.3:
                    if gt_exists_detection[index_max_overlap_bbox] == 0:
                        true_positive[detection_idx] = 1
                        gt_exists_detection[index_max_overlap_bbox] = 1
                        f.write(f" TP: gt_bbox: {gt_detections_on_image[best_gt_local]}, bbox: {detections[detection_idx]}, iou: {max_overlap}, score: {scores[detection_idx]}")
                    else:
                        false_positive[detection_idx] = 1
                        duplicated_detections[detection_idx] = 1
                        f.write(f" TP (dar duplicat): gt_bbox: {gt_detections_on_image[best_gt_local]}, bbox: {detections[detection_idx]}, iou: {max_overlap}, score: {scores[detection_idx]}")

                else:
                    false_positive[detection_idx] = 1
                    f.write(f" FP (iou prea mic): gt_bbox: {gt_detections_on_image[best_gt_local]}, bbox: {detections[detection_idx]}, iou: {max_overlap}, score: {scores[detection_idx]}")


            cum_false_positive = np.cumsum(false_positive)
            cum_true_positive = np.cumsum(true_positive)

            rec = cum_true_positive / num_gt_detections
            
            #f.write(f"\n\n recall: {rec}")
            prec = cum_true_positive / (cum_true_positive + cum_false_positive)
            #f.write(f"prec: {prec}")
            average_precision = self.compute_average_precision(rec, prec)
            plt.plot(rec, prec, '-')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Average precision %.3f' % average_precision)
            plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
            plt.show()
            
            f.write("===== EVALUATION DEBUG =====\n\n")

            # ---- numere globale ----
            f.write(f"Num GT detections        : {num_gt_detections}\n")
            f.write(f"Num predicted detections : {num_detections}\n")
            f.write(f"True Positives total     : {int(np.sum(true_positive))}\n")
            f.write(f"False Positives total    : {int(np.sum(false_positive))}\n")
            f.write(f"Duplicated detections    : {int(np.sum(duplicated_detections))}\n\n")

            # ---- recall / precision ----
            f.write(f"Max recall               : {rec.max():.4f}\n")
            f.write(f"Precision @ max recall   : {prec[-1]:.4f}\n")

            # precision la cateva praguri de recall
            for r_thr in [0.05, 0.1, 0.2, 0.3, 0.5]:
                idx = np.searchsorted(rec, r_thr)
                if idx < len(prec):
                    f.write(f"Precision @ recall {r_thr:.2f} : {prec[idx]:.4f}\n")

            f.write("\n")

            # ---- scoruri TP / FP ----
            tp_scores = scores[true_positive == 1]
            fp_scores = scores[false_positive == 1]

            if len(tp_scores) > 0:
                f.write("TP scores:\n")
                f.write(f"  count  : {len(tp_scores)}\n")
                f.write(f"  min    : {tp_scores.min():.4f}\n")
                f.write(f"  mean   : {tp_scores.mean():.4f}\n")
                f.write(f"  median : {np.median(tp_scores):.4f}\n")
                f.write(f"  max    : {tp_scores.max():.4f}\n\n")

            if len(fp_scores) > 0:
                f.write("FP scores:\n")
                f.write(f"  count  : {len(fp_scores)}\n")
                f.write(f"  min    : {fp_scores.min():.4f}\n")
                f.write(f"  mean   : {fp_scores.mean():.4f}\n")
                f.write(f"  median : {np.median(fp_scores):.4f}\n")
                f.write(f"  max    : {fp_scores.max():.4f}\n\n")

            # ---- average precision ----
            f.write(f"Average Precision (AP)   : {average_precision:.4f}\n")
