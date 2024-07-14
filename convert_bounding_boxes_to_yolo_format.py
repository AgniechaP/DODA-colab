import os

def convert_to_yolo_format(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max, _ = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return f"0 {x_center} {y_center} {width} {height}"

def process_input_file(input_file_path, output_dir, img_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            img_path = parts[0]
            bboxes = parts[1:]

            img_name = os.path.basename(img_path)
            img_id = os.path.splitext(img_name)[0]
            output_file_path = os.path.join(output_dir, f"{img_id}.txt")

            with open(output_file_path, 'w') as output_file:
                for bbox in bboxes:
                    bbox_coords = list(map(int, bbox.split(',')))
                    yolo_bbox = convert_to_yolo_format(bbox_coords, img_size, img_size)
                    output_file.write(yolo_bbox + '\n')

input_file_path = '/content/drive/MyDrive/DODA/v2/random_layout/10_07_random_layout/bounding_boxes.txt'
output_dir = '/content/drive/MyDrive/DODA/v2/random_layout/10_07_random_layout/yolo_labels'
img_size = 512

process_input_file(input_file_path, output_dir, img_size)