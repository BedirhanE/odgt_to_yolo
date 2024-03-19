import os

def convert_odgt_to_yolo(annotations_path, output_dir):
    with open(annotations_path, 'r') as file: #dosyayı okuma modunda açtım
        lines = file.readlines()

    for line in lines:  # her bir satırı incelemek için döngü kullandım
        data = eval(line) #her bir satır içeriği inceleme yapmak için eval fonk. kullandım.
        img_id = data["ID"] #her bir satırın içindeki görüntü kimlikleriini aldım.

        #görüntü genişlik ve yükseklik hesabı
        img_width, img_height = calculate_image_size(data["gtboxes"])

        label_content = ""
        for box in data["gtboxes"]:
            if box['tag'] == 'person':
                bbox = box["hbox"]
                label_content += f"0 {(bbox[0] + bbox[2] / 2) / img_width} {(bbox[1] + bbox[3] / 2) / img_height} {bbox[2] / img_width} {bbox[3] / img_height}\n"

        with open(os.path.join(output_dir, img_id + '.txt'), 'w') as label_file:
            label_file.write(label_content)

def calculate_image_size(boxes):
    xmin = min(box["hbox"][0] for box in boxes)
    xmax = max(box["hbox"][0] + box["hbox"][2] for box in boxes)
    ymin = min(box["hbox"][1] for box in boxes)
    ymax = max(box["hbox"][1] + box["hbox"][3] for box in boxes)

    width = xmax - xmin
    height = ymax - ymin

    return width, height


annotations_path = "data/annotation_train.odgt"
output_dir = "data/yolo_labels"  #yolo formatına donş. etiketlerin kaydedileceği klasör
convert_odgt_to_yolo(annotations_path, output_dir)
