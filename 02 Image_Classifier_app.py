
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# python predict.py ./test_images/cautleya_spicata.jpg my_model.h5
# python predict.py ./test_images/cautleya_spicata.jpg my_model.h5 --top_k 3
# python predict.py ./test_images/cautleya_spicata.jpg my_model.h5 --category_names label_map.json
class Classifier:
    def __init__(self, image_path, model_name, top_k, json_name):
        self.image_path = image_path
        self.model_name = model_name
        self.top_k = top_k if top_k != None else 1
        self.json_name = json_name if json_name != None else 'label_map.json'
        self.image_size = 224
        
    def __load_model(self):
        model = tf.keras.models.load_model(self.model_name, custom_objects={ 'KerasLayer': hub.KerasLayer },
                                          compile=False)
        return model
    def __load_class_names(self):
        with open(self.json_name, 'r') as f:
            class_names = json.load(f)
        return class_names
    def __process_image(self, image):
        image_tensor = tf.convert_to_tensor(image)
        image = tf.image.resize(image_tensor, (self.image_size, self.image_size))
        image /= 255
        return image
    def __convert_image(self):
        image_obj = Image.open(self.image_path)
        image_array = np.asarray(image_obj)
        expanded_img = np.expand_dims(self.__process_image(image_array), axis=0)
        return expanded_img        
    def predict(self):
        expanded_img = self.__convert_image()
        model = self.__load_model()
        print('MODEL:::', model)
        predictions = model.predict(expanded_img)
        values, indices = tf.nn.top_k(predictions, k=self.top_k)
        indices = 1 + indices[0]
        
        class_names = self.__load_class_names()
        converted_indices = indices.numpy().astype(str)
        classes_with_name = [class_names[x] for x in converted_indices]
        return classes_with_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('model_name')
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--category_names')
    
    args = parser.parse_args()
#     print(args)
    classifier = Classifier(args.image_path, args.model_name, args.top_k, args.category_names)
    print(">>> PREDICTED FLOWER NAME: ", classifier.predict())