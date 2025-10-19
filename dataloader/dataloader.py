import numpy as np
import os
import pickle
from enum import Enum
from PIL import Image
from typing import Dict, Any, Tuple, Iterator, Optional


class DatasetType(Enum):
    BBDD = "BBDD"
    QSD1_W1 = "qsd1_w1"
    QST1_W1 = "qst1_w1"
    QSD2_W2 = "qsd2_w2"
    QST1_W2 = "qst1_w2"
    QST2_W2 = "qst2_w2"
    QSD1_W3 = "qsd1_w3"
    QSD2_W3 = "qsd2_w3"


class WeekFolder(Enum):
    WEEK_1 = "week_1"
    WEEK_2 = "week_2"
    WEEK_3 = "week_3"


class DataLoader:
    """Load and manage BBDD and qsd1_w1 computer vision datasets."""

    def __init__(self):
        current_file = os.path.abspath(__file__)
        self.root_path = os.path.dirname(os.path.dirname(current_file))
        self.data_path = os.path.join(self.root_path, "data")
        self.data: Dict[int, Dict[str, Any]] = {}
        self.dataset_type: Optional[DatasetType] = None

    def _reverse_dict(self, d: Dict[Any, Any]) -> Dict[Any, Any]:
        return {value: key for key, value in d.items()}
    
    def has_ground_truth(self) -> bool:
        if self.dataset_type is None:
            return False
        # Only validation datasets have ground truth, test datasets (QST*) do not
        return self.dataset_type in [DatasetType.BBDD, DatasetType.QSD1_W1, DatasetType.QSD2_W2, DatasetType.QSD1_W3, DatasetType.QSD2_W3]

    def load_dataset(self, dataset: DatasetType) -> None:
        """Load dataset by DatasetType enum."""
        self.clear_dataset()

        if dataset == DatasetType.BBDD:
            self.load_BBDD()
        elif dataset == DatasetType.QSD1_W1:
            self.load_qsd1_w1()
        elif dataset == DatasetType.QST1_W1:
            self.load_qst1_w1()
        elif dataset == DatasetType.QSD2_W2:
            self.load_qsd2_w2()
        elif dataset == DatasetType.QST1_W2:
            self.load_qst1_w2()
        elif dataset == DatasetType.QST2_W2:
            self.load_qst2_w2()
        elif dataset == DatasetType.QSD1_W3:
            self.load_qsd1_w3()
        elif dataset == DatasetType.QSD2_W3:
            self.load_qsd2_w3()

        self.dataset_type = dataset

    def load_BBDD(self) -> None:
        """Load BBDD dataset: JPG images, TXT metadata, and relationships.pkl."""
        dataset_path = os.path.join(self.data_path, "BBDD")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"BBDD dataset path not found: {dataset_path}")

        relationships_file = os.path.join(dataset_path, "relationships.pkl")
        if not os.path.exists(relationships_file):
            raise FileNotFoundError(
                f"Relationships file not found: {relationships_file}"
            )

        try:
            with open(relationships_file, "rb") as f:
                relationships = pickle.load(f)
            relationships = self._reverse_dict(relationships)
        except Exception as e:
            raise Exception(f"Error loading relationships: {e}")

        try:
            files = [
                f
                for f in os.listdir(dataset_path)
                if os.path.isfile(os.path.join(dataset_path, f))
                and not f.endswith(".pkl")
            ]
            names = set(f.split(".")[0] for f in files)
        except Exception as e:
            raise Exception(f"Error reading dataset directory: {e}")

        for name in names:
            try:
                if not name.startswith("bbdd_"):
                    continue

                id_str = name.split("_")[1]
                image_id = int(id_str)

                jpg_filename = os.path.join(dataset_path, f"{name}.jpg")
                if not os.path.exists(jpg_filename):
                    print(f"Warning: JPG file not found for {name}, skipping...")
                    continue

                image = np.array(Image.open(jpg_filename))

                txt_filename = os.path.join(dataset_path, f"{name}.txt")
                if not os.path.exists(txt_filename):
                    info = ""
                else:
                    try:
                        with open(
                            txt_filename, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            info = f.readline().strip()
                    except Exception:
                        try:
                            with open(txt_filename, "r", encoding="latin-1") as f:
                                info = f.readline().strip()
                        except Exception:
                            info = ""

                relationship = relationships.get(image_id, None)

                self.data[image_id] = {
                    "image": image,
                    "info": info,
                    "relationship": relationship,
                }

            except Exception as e:
                print(f"Warning: Error processing {name}: {e}")
                continue

        print(f"Successfully loaded {len(self.data)} images from BBDD dataset")

    def load_qsd1_w1(self) -> None:
        """Load qsd1_w1 dataset: JPG images and gt_corresps.pkl."""
        dataset_path = os.path.join(self.data_path, "qsd1_w1")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"qsd1_w1 dataset path not found: {dataset_path}")

        gt_file = os.path.join(dataset_path, "gt_corresps.pkl")
        gt_correspondences = []

        if os.path.exists(gt_file):
            try:
                with open(gt_file, "rb") as f:
                    gt_correspondences = pickle.load(f)
            except Exception as e:
                print(f"Warning: Error loading ground truth correspondences: {e}")

        try:
            files = [
                f
                for f in os.listdir(dataset_path)
                if f.endswith(".jpg") and os.path.isfile(os.path.join(dataset_path, f))
            ]

            for filename in files:
                try:
                    name_without_ext = filename.split(".")[0]
                    image_id = int(name_without_ext)

                    jpg_filename = os.path.join(dataset_path, filename)
                    image = np.array(Image.open(jpg_filename))

                    gt_correspondence = (
                        gt_correspondences[image_id]
                        if image_id < len(gt_correspondences)
                        else None
                    )

                    self.data[image_id] = {
                        "image": image,
                        "info": f"Query image {name_without_ext}",
                        "relationship": gt_correspondence,
                    }

                except Exception as e:
                    print(f"Warning: Error processing {filename}: {e}")
                    continue

        except Exception as e:
            raise Exception(f"Error reading qsd1_w1 directory: {e}")

        print(f"Successfully loaded {len(self.data)} images from qsd1_w1 dataset")

    def load_qst1_w1(self) -> None:
        """Load qst1_w1 dataset: JPG images."""
        dataset_path = os.path.join(self.data_path, "qst1_w1")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"qst1_w1 dataset path not found: {dataset_path}")

        try:
            files = [
                f
                for f in os.listdir(dataset_path)
                if f.endswith(".jpg") and os.path.isfile(os.path.join(dataset_path, f))
            ]

            for filename in files:
                try:
                    name_without_ext = filename.split(".")[0]
                    image_id = int(name_without_ext)

                    jpg_filename = os.path.join(dataset_path, filename)
                    image = np.array(Image.open(jpg_filename))

                    self.data[image_id] = {
                        "image": image,
                        "info": f"Query image {name_without_ext}",
                        "relationship": None,
                    }

                except Exception as e:
                    print(f"Warning: Error processing {filename}: {e}")
                    continue

        except Exception as e:
            raise Exception(f"Error reading qst1_w1 directory: {e}")

        print(f"Successfully loaded {len(self.data)} images from qst1_w1 dataset")

    def load_qsd2_w2(self) -> None:
        """Load qsd2_w2 dataset: JPG images (original) and PNG images (background removed) with gt_corresps.pkl."""
        dataset_path = os.path.join(self.data_path, "qsd2_w2")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"qsd2_w2 dataset path not found: {dataset_path}")

        gt_file = os.path.join(dataset_path, "gt_corresps.pkl")
        gt_correspondences = []

        if os.path.exists(gt_file):
            try:
                with open(gt_file, "rb") as f:
                    gt_correspondences = pickle.load(f)
            except Exception as e:
                print(f"Warning: Error loading ground truth correspondences: {e}")

        try:
            # Get all JPG files (original images)
            jpg_files = [
                f
                for f in os.listdir(dataset_path)
                if f.endswith(".jpg") and os.path.isfile(os.path.join(dataset_path, f))
            ]

            for jpg_filename in jpg_files:
                try:
                    name_without_ext = jpg_filename.split(".")[0]
                    image_id = int(name_without_ext)

                    # Load original JPG image
                    jpg_path = os.path.join(dataset_path, jpg_filename)
                    original_image = np.array(Image.open(jpg_path))

                    # Load corresponding PNG image (background removed)
                    png_filename = f"{name_without_ext}.png"
                    png_path = os.path.join(dataset_path, png_filename)
                    background_removed_image = None
                    
                    if os.path.exists(png_path):
                        background_removed_image = np.array(Image.open(png_path))
                    else:
                        print(f"Warning: PNG file not found for {name_without_ext}")

                    gt_correspondence = (
                        gt_correspondences[image_id]
                        if image_id < len(gt_correspondences)
                        else None
                    )

                    self.data[image_id] = {
                        "image": original_image,  # Original image with background
                        "background_removed": background_removed_image,  # Image with background removed
                        "info": f"Query image {name_without_ext} (original + background removed)",
                        "relationship": gt_correspondence,
                    }

                except Exception as e:
                    print(f"Warning: Error processing {jpg_filename}: {e}")
                    continue

        except Exception as e:
            raise Exception(f"Error reading qsd2_w2 directory: {e}")

        print(f"Successfully loaded {len(self.data)} images from qsd2_w2 dataset (with both original and background-removed versions)")

    def load_qsd1_w3(self) -> None:
        """Load qsd1_w3 dataset: JPG images, non-augmented JPG images and gt_corresps.pkl. (masks irrelevant)"""
        dataset_path = os.path.join(self.data_path, "qsd1_w3")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"qsd1_w3 dataset path not found: {dataset_path}")

        gt_file = os.path.join(dataset_path, "gt_corresps.pkl")
        gt_correspondences = []

        if os.path.exists(gt_file):
            try:
                with open(gt_file, "rb") as f:
                    gt_correspondences = pickle.load(f)
            except Exception as e:
                print(f"Warning: Error loading ground truth correspondences: {e}")

        try:
            files = [
                f
                for f in os.listdir(dataset_path)
                if f.endswith(".jpg") and os.path.isfile(os.path.join(dataset_path, f))
            ]

            for filename in files:
                try:
                    name_without_ext = filename.split(".")[0]
                    image_id = int(name_without_ext)

                    jpg_filename = os.path.join(dataset_path, filename)
                    non_augm_jpg_filename = os.path.join(dataset_path, "non_augmented", filename)
                    image = np.array(Image.open(jpg_filename))
                    non_augm_image = np.array(Image.open(non_augm_jpg_filename))

                    gt_correspondence = (
                        gt_correspondences[image_id]
                        if image_id < len(gt_correspondences)
                        else None
                    )

                    txt_filename = os.path.join(dataset_path, f"{name_without_ext}.txt")
                    with open(txt_filename, "r", encoding="utf-8", errors="ignore") as f:
                        info = f.readline().strip()

                    self.data[image_id] = {
                        "image": image,
                        "non_augm_image": non_augm_image,
                        "info": info,
                        "relationship": gt_correspondence,
                    }

                except Exception as e:
                    print(f"Warning: Error processing {filename}: {e}")
                    continue

        except Exception as e:
            raise Exception(f"Error reading qsd1_w3 directory: {e}")

        print(f"Successfully loaded {len(self.data)} images from qsd1_w3 dataset")

    def load_qsd2_w3(self) -> None:
        """Load qsd2_w3 dataset: JPG images, non-augmented JPG images and gt_corresps.pkl. (PNG and TXT files ignored)"""
        dataset_path = os.path.join(self.data_path, "qsd2_w3")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"qsd2_w3 dataset path not found: {dataset_path}")

        gt_file = os.path.join(dataset_path, "gt_corresps.pkl")
        gt_correspondences = []

        if os.path.exists(gt_file):
            try:
                with open(gt_file, "rb") as f:
                    gt_correspondences = pickle.load(f)
            except Exception as e:
                print(f"Warning: Error loading ground truth correspondences: {e}")

        try:
            files = [
                f
                for f in os.listdir(dataset_path)
                if f.endswith(".jpg") and os.path.isfile(os.path.join(dataset_path, f))
            ]

            for filename in files:
                try:
                    name_without_ext = filename.split(".")[0]
                    image_id = int(name_without_ext)

                    jpg_filename = os.path.join(dataset_path, filename)
                    non_augm_jpg_filename = os.path.join(dataset_path, "non_augmented", filename)
                    image = np.array(Image.open(jpg_filename))
                    non_augm_image = np.array(Image.open(non_augm_jpg_filename))

                    gt_correspondence = (
                        gt_correspondences[image_id]
                        if image_id < len(gt_correspondences)
                        else None
                    )

                    txt_filename = os.path.join(dataset_path, f"{name_without_ext}.txt")
                    with open(txt_filename, "r", encoding="utf-8", errors="ignore") as f:
                        info = f.readline().strip()

                    self.data[image_id] = {
                        "image": image,
                        "non_augm_image": non_augm_image,
                        "info": info,
                        "relationship": gt_correspondence,
                    }

                except Exception as e:
                    print(f"Warning: Error processing {filename}: {e}")
                    continue

        except Exception as e:
            raise Exception(f"Error reading qsd2_w3 directory: {e}")

        print(f"Successfully loaded {len(self.data)} images from qsd2_w3 dataset")

    def load_qst1_w2(self) -> None:
        """Load qst1_w2 dataset: JPG images."""
        dataset_path = os.path.join(self.data_path, "qst1_w2")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"qst1_w2 dataset path not found: {dataset_path}")

        try:
            files = [
                f
                for f in os.listdir(dataset_path)
                if f.endswith(".jpg") and os.path.isfile(os.path.join(dataset_path, f))
            ]

            for filename in files:
                try:
                    name_without_ext = filename.split(".")[0]
                    image_id = int(name_without_ext)

                    jpg_filename = os.path.join(dataset_path, filename)
                    image = np.array(Image.open(jpg_filename))

                    self.data[image_id] = {
                        "image": image,
                        "info": f"Query image {name_without_ext}",
                        "relationship": None,
                    }

                except Exception as e:
                    print(f"Warning: Error processing {filename}: {e}")
                    continue

        except Exception as e:
            raise Exception(f"Error reading qst1_w2 directory: {e}")

        print(f"Successfully loaded {len(self.data)} images from qst1_w2 dataset")

    def load_qst2_w2(self) -> None:
        """Load qst2_w2 dataset: JPG images."""
        dataset_path = os.path.join(self.data_path, "qst2_w2")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"qst2_w2 dataset path not found: {dataset_path}")

        try:
            files = [
                f
                for f in os.listdir(dataset_path)
                if f.endswith(".jpg") and os.path.isfile(os.path.join(dataset_path, f))
            ]

            for filename in files:
                try:
                    name_without_ext = filename.split(".")[0]
                    image_id = int(name_without_ext)

                    jpg_filename = os.path.join(dataset_path, filename)
                    image = np.array(Image.open(jpg_filename))

                    self.data[image_id] = {
                        "image": image,
                        "info": f"Query image {name_without_ext}",
                        "relationship": None,
                    }

                except Exception as e:
                    print(f"Warning: Error processing {filename}: {e}")
                    continue

        except Exception as e:
            raise Exception(f"Error reading qst2_w2 directory: {e}")

        print(f"Successfully loaded {len(self.data)} images from qst2_w2 dataset")

    def clear_dataset(self) -> None:
        self.data = {}
        self.dataset_type = None

    def iterate_images(self) -> Iterator[Tuple[int, np.ndarray, str, Any]]:
        """
        Yield (id, image, info, relationship) for each loaded image.
        If dataset has non augmented image, yield
              (id, image, non_augm_iamge, info, relationship).
        """
        for image_id, values in self.data.items():
            if self.dataset_type in [DatasetType.QSD1_W3, DatasetType.QSD2_W3]:
                yield (
                    image_id,
                    values["image"],
                    values["non_augm_image"],
                    values["info"],
                    values["relationship"],
                )
            else:
                yield (
                    image_id,
                    values["image"],
                    values["info"],
                    values["relationship"],
            )
                
    def get_image_by_id(self, image_id: int) -> Optional[Dict[str, Any]]:
        return self.data.get(image_id)

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "dataset_type": self.dataset_type.value if self.dataset_type else None,
            "num_images": len(self.data),
            "image_ids": list(self.data.keys()) if self.data else [],
            "data_path": self.data_path,
        }
