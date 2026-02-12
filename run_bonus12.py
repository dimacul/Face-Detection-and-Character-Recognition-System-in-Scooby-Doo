import numpy as np
import os
from ultralytics import YOLO

#-------------------------------------------------------------------------------------------------------------------------
# INLOCUITI CU CALEA (RELATIVA LA FOLDERUL ROOT 341_Dima_Cristian) CATRE FOLDERUL CU IMAGINILE DE TEST

test_images_path = "../testare"

#-------------------------------------------------------------------------------------------------------------------------


def load_scooby_model(model_path='yolov8n.pt'):
    model = YOLO(model_path)
    return model

# def train_scooby_model(model):
#     model.train(data='yolo_config.yaml', epochs=50, imgsz=480, 
#                 device="cpu", 
#                 batch=-1, 
#                 workers=12, 
#                 patience=20,
#                 name='yolo_finetuning_scooby')


def run_inference_and_save_results(model, test_images_path):
    
    t1_detections = []
    t1_scores = []
    t1_filenames = []

    characters = ['fred', 'daphne', 'shaggy', 'velma']
    t2_data = {char: {'det': [], 
                        'scr': [], 
                        'fn': []
                    } for char in characters}

    mapping = {0: 'fred', 1: 'daphne', 2: 'shaggy', 3: 'velma', 4: 'unknown'}

    image_files = sorted([f for f in os.listdir(test_images_path) if f.endswith('.jpg')])

    print(f"YOLO - inferenta pe {len(image_files)} imagini")

    for img_name in image_files:
        img_path = os.path.join(test_images_path, img_name)
        results = model.predict(source=img_path, conf=0.01, verbose=False)

        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy() # [xmin, ymin, xmax, ymax]
                score = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                char_name = mapping.get(cls_id, 'unknown')

                
                t1_detections.append(coords) # pt task 1
                t1_scores.append(score)
                t1_filenames.append(img_name)

                
                if char_name in characters: # pt task 2
                    t2_data[char_name]['det'].append(coords)
                    t2_data[char_name]['scr'].append(score)
                    t2_data[char_name]['fn'].append(img_name)

    # SALVEZ TASK 1
    np.save('fisiereSolutie/341_Dima_Cristian/bonus/task1/detections_all_faces.npy', np.array(t1_detections))
    np.save('fisiereSolutie/341_Dima_Cristian/bonus/task1/scores_all_faces.npy', np.array(t1_scores))
    np.save('fisiereSolutie/341_Dima_Cristian/bonus/task1/file_names_all_faces.npy', np.array(t1_filenames))
    print(f"Task 1 bonus yolo: {len(t1_scores)} detectii all faces.")

    # SLAVEZ TASK 2
    for char in characters:
        np.save(f'fisiereSolutie/341_Dima_Cristian/bonus/task2/detections_{char}.npy', np.array(t2_data[char]['det']))
        np.save(f'fisiereSolutie/341_Dima_Cristian/bonus/task2/scores_{char}.npy', np.array(t2_data[char]['scr']))
        np.save(f'fisiereSolutie/341_Dima_Cristian/bonus/task2/file_names_{char}.npy', np.array(t2_data[char]['fn']))
        print(f"Task 2 - {char} salvat: {len(t2_data[char]['scr'])} detectii")

    print("\n[SUCCESS] Toate fisierele .npy au fost generate")


model = load_scooby_model('yolo_finetunat/best.pt') #yolov8n.pt   #'runs/detect/yolo_finetuning_scooby2/weights/best.pt'
print(model.names)
#train_scooby_model(model)

run_inference_and_save_results(model, test_images_path)