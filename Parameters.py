import os

class Parameters:
    def __init__(self):
        self.base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__))) #, 'data'
        # self.dir_pos_examples = os.path.join(self.base_dir, 'exemplePozitive')
        # self.dir_neg_examples = os.path.join(self.base_dir, 'exempleNegative2')
        self.dir_test_examples = os.path.join(self.base_dir, '../validare/validare')  # 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        #self.path_annotations = os.path.join(self.base_dir, '../../validare/task1_gt_validare.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'fisiereSolutie/341_Dima_Cristian')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))
            
        self.dir_load_model_task1 = 'modeleSVM/task1'
        self.dir_load_model_task2 = 'modeleSVM/task2'

        # set the parameters
        self.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 13086  # numarul exemplelor pozitive
        self.number_negative_examples = 40000  # numarul exemplelor negative
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0

        print("base_dir:", os.path.abspath(self.base_dir))
        # print("dir_pos_examples:", os.path.abspath(self.dir_pos_examples))
        # print("dir_neg_examples:", os.path.abspath(self.dir_neg_examples))
        print("dir_test_examples:", os.path.abspath(self.dir_test_examples))
        # print("path_annotations:", os.path.abspath(self.path_annotations))
        print("dir_save_files:", os.path.abspath(self.dir_save_files))
        print("dir_load_model_task1:", os.path.abspath(self.dir_load_model_task1))
        print("dir_load_model_task2:", os.path.abspath(self.dir_load_model_task2))

        

