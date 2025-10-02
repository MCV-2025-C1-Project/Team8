import numpy as np
import os
import pickle
from PIL import Image


class DataLoader():
    def __init__(self):
        current_file = os.path.abspath(__file__)
        self.root_path = os.path.dirname(os.path.dirname(current_file))
        self.data_path = os.path.join(self.root_path, "data")
        self.data = {}
    
    def reverse_dict(self, d):
        reversed_d = {}
        for key, value in d.items():
            reversed_d[value] = key
        return reversed_d

    def load_dataset(self, dataset):
        """
        Args:
            dataset (string): "BBDD" or "qsd1_w1"
        """
        self.clear_dataset()
        assert dataset in ["BBDD", "qsd1_w1"]
        
        dataset_path = os.path.join(self.data_path, dataset)
        files = [f for f in os.listdir(dataset_path) \
                 if os.path.isfile(os.path.join(dataset_path, f))]
        names = set([f.split(".")[0] for f in files])
        
        if dataset == "BBDD":
            # Load relationships
            with open(os.path.join(dataset_path, "relationships.pkl"), 'rb') as f:
                relationships = pickle.load(f)
            relationships = self.reverse_dict(relationships)
            names.remove("relationships")
            # Load images
            for name in names:
                id = int(name.split("_")[1])
                png_filename = os.path.join(dataset_path, name+".png")
                image = np.array(Image.open(png_filename))
                txt_filename = os.path.join(dataset_path, name+".txt")
                with open(txt_filename, "r") as f:
                    info = f.readline().rstrip("\n")
                
                self.data[id] = {
                    "image": image,
                    "info": info,
                    "relationship": relationships[id],
                }

        elif dataset == "qsd1_w1":
            pass

    def clear_dataset(self):
        self.data = {}
    
    def iterate_images(self):
        """
        Iterate over all images in the dataset.
        Yields:
            tuple: (id, image, info, relationship)
        """
        for id, values in self.data.items():
            yield id, values["image"], values["info"], values["relationship"]


if __name__ == "__main__":
    dl = DataLoader()
    print(dl.root_path)
    dl.load_dataset(dataset="BBDD")
    for id, image, info, relationship in dl.iterate_images():
        print(id, image.shape, info, relationship)