from dataloader.dataloader import DataLoader, DatasetType, WeekFolder
from preprocessing.preprocessors import PreprocessingMethod

import cv2 as cv
from tqdm import tqdm as TQDM
from typing import Dict
import numpy as np


class KeyPointImageRetrievalSystem:
    def __init__(self):
        self.index_dataset = DataLoader()
        self.query_dataset = DataLoader()

        self.index_descriptors: Dict[int, tuple] = {} # image_id -> (keypoints, descriptors)
        self.query_descriptors: Dict[int, tuple] = {} # image_id -> (keypoints, descriptors)

        self.ground_truth = []

    def load_ground_truth(self) -> None:
        """Load ground truth correspondences for QSD2_W2."""
        self.ground_truth = []
        for *_, relationship in self.query_dataset.iterate_images():
            if relationship is not None:
                if isinstance(relationship, list):
                    self.ground_truth.append(relationship)
                else:
                    self.ground_truth.append([relationship])
            else:
                self.ground_truth.append([])
        print(f"Loaded ground truth for {len(self.ground_truth)} query images")

    def compute_keypoint_descriptors(self, method):
        self.index_descriptors.clear()
        self.query_descriptors.clear()

        # Query descriptors
        progress_bar = TQDM(
            self.query_dataset.iterate_images(),
            total=len(self.query_dataset.data),
            desc="Computing query descriptors"
        )

        for image_id, image, *_ in progress_bar:
            keypoints, descriptors = method.compute(
                image,
            )
            self.query_descriptors[image_id] = (keypoints, descriptors)

        # Index descriptors
        progress_bar = TQDM(
            self.index_dataset.iterate_images(),
            total=len(self.index_dataset.data),
            desc="Computing index descriptors"
        )

        for image_id, image, *_ in progress_bar:
            keypoints, descriptors = method.compute(
                image,
            )
            self.index_descriptors[image_id] = (keypoints, descriptors)
    
    def retrieve(
            self,
            query_keypoints,
            query_descriptors,
            n=5, # images to retrieve
            norm_type=cv.NORM_HAMMING,
            ratio_threshold=0.75,
            min_matches=10,
        ):
        matcher = cv.BFMatcher(normType=norm_type)

        results = []
        for image_id, *_ in self.index_dataset.iterate_images():
            keypoints, descriptors = self.index_descriptors[image_id]

            # Checks
            assert descriptors is not None and query_descriptors is not None, \
                f"One or both descriptors are None. \
                    Got type(descriptors)={type(descriptors)}, \
                          type(query_descriptors)={type(query_descriptors)}"
            if descriptors.dtype != query_descriptors.dtype:
                # ORB produces uint8, other methods produce float32
                descriptors = descriptors.astype(np.float32)
                query_descriptors = query_descriptors.astype(np.float32)
            assert descriptors.shape[1] == query_descriptors.shape[1]
            
            # Match
            knn_matches = matcher.knnMatch(query_descriptors, descriptors, k=2)
            if len(knn_matches) == 0:
                continue

            # Apply Lowe’s ratio test to filter out poor matches
            # https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
            good_matches = []
            for m, n in knn_matches:
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
            if len(good_matches) < min_matches:
                continue

            # Verify spatial consistency using homography
            src_pts = np.float32(
                [query_keypoints[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [keypoints[m.trainIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            if H is None:
                continue
            n_inliers = int(mask.sum())
            matches_mask = mask.ravel().tolist()
            score = sum(matches_mask) / len(good_matches)

            # Keep results
            results.append({
                "image_id": image_id,
                "num_good_matches": len(good_matches),
                "n_inliers": n_inliers,
                "score": score,
                "homography": H,
                "good_matches": good_matches,
            })

        # TODO: per cada imatge, en funció dels results, retornar els "n"
        # matches o retornar [-1] si es considera que no en té cap
        # (s'ha de retornar llista de llistes)

    def run(
        self,
        local_descriptor_method,
        index_dataset: DatasetType,
        query_dataset: DatasetType,
        week_folder: WeekFolder,
        preprocessing: PreprocessingMethod = None, # Fer-ho bé
        save_results: bool = True,
    ):
        
        # Load datasets
        self.index_dataset.load_dataset(index_dataset)
        self.query_dataset.load_dataset(query_dataset)

        # Only load ground truth if the dataset has it
        if self.query_dataset.has_ground_truth():
            self.load_ground_truth()
        
        # Descriptors
        self.compute_keypoint_descriptors(
            local_descriptor_method,
        )

        # Retrieve
        for image_id, *_ in self.query_dataset.iterate_images():
            image_kp, image_dsc = self.query_descriptors[image_id]
            matches = self.retrieve(
                query_keypoints=image_kp,
                query_descriptors=image_dsc,
                n=5,
            )
        
