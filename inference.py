import argparse
import os
import glob
import json
import tqdm
import numpy as np
from PIL import Image
from tqdm import tqdm
from vaik_segmentation_tflite_inference.tflite_model import TfliteModel


def main(input_saved_model_path, input_classes_path, input_image_dir_path, answer_image_dir_path, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)
    with open(input_classes_path, 'r') as f:
        classes = f.readlines()
    classes = tuple([label.strip() for label in classes])

    model = TfliteModel(input_saved_model_path, classes)

    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_path_list = []
    for file in types:
        image_path_list.extend(glob.glob(os.path.join(input_image_dir_path, f'{file}'), recursive=True))
    image_list = []
    for image_path in image_path_list:
        image = np.asarray(Image.open(image_path).convert('RGB'))
        image_list.append(image)

    import time
    total_time = 0
    output_list = []
    for image in tqdm(image_list, desc='inference'):
        start = time.time()
        output, raw_pred = model.inference(image)
        end = time.time()
        total_time += end - start
        output_list.append(output)

    for image_path, output_elem in zip(image_path_list, output_list):
        answer_image = np.asarray(
            Image.open(os.path.join(answer_image_dir_path, os.path.basename(image_path).replace('raw', 'seg'))).convert(
                'L'))
        output_json_path = os.path.join(output_dir_path, os.path.splitext(os.path.basename(image_path))[0] + '.json')
        output_elem['answer'] = {'array': answer_image.flatten().tolist(), 'shape': answer_image.shape}
        output_elem['labels'] = {'array': output_elem['labels'].flatten().tolist(),
                                 'shape': output_elem['labels'].shape}
        with open(output_json_path, 'w') as f:
            json.dump(output_elem, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
    print(f'{len(image_list) / (end - start)}[images/sec]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--input_saved_model_path', type=str,
                        default='~/.vaik-segmentation-tflite-exporter/model.tflite')
    parser.add_argument('--input_classes_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'test_images/classes.txt'))
    parser.add_argument('--input_image_dir_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'test_images/raw'))
    parser.add_argument('--answer_image_dir_path', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'test_images/seg'))
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik-segmentation-tflite-experiment/test_images_out')
    args = parser.parse_args()

    args.input_saved_model_path = os.path.expanduser(args.input_saved_model_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)
    args.input_image_dir_path = os.path.expanduser(args.input_image_dir_path)
    args.answer_image_dir_path = os.path.expanduser(args.answer_image_dir_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    main(**args.__dict__)
