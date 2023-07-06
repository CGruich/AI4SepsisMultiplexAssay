import numpy as np
import os.path
import json
from object_detection import RegionDetector
import cv2
import bayes_opt
from bayes_opt import SequentialDomainReductionTransformer

class MSEROptimizer(object):
    def __init__(self, normalized_images, particle_locations, num_iterations):
        self.images = normalized_images
        self.particle_locations = particle_locations

        for i in range(len(self.images)):
            if self.images[i].dtype == 'uint16':
                img = np.divide(self.images[i], 256)
                img = img.astype('uint8')
                self.images[i] = img

        self.detector = RegionDetector(None)
        self.max_area = 4200
        self.min_area = 3500
        self.best_score = -np.inf
        self.save_filename = None
        self.num_iterations = num_iterations

    def train(self, save_directory: str = None):
        if save_directory is not None:
            self.save_filename = save_directory
        
        ranges = {
            "delta": (1, 30),
            "min_area": (100, 2000),
            "max_area": (500, 2000),
            "max_variation": (0, 1),
            "min_diversity": (0, 1),
            "max_evolution": (0, 1000),
            "area_threshold": (0, 1000),
            "min_margin": (0, 1),
            "edge_blur_size": (0, 1000)}

        optimizer = bayes_opt.BayesianOptimization(self.evaluate, ranges, verbose=1)
        optimizer.maximize(init_points=100, n_iter=self.num_iterations)
        
        if self.save_filename is not None:
            saveDict = {"optimizer.max": optimizer.max}
            # Save .json file
            with open(self.save_filename, "w") as jsonFile:
                json.dump(saveDict, jsonFile)

    def evaluate(self, delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold, min_margin, edge_blur_size):
        self.detector.MSER_parameters = (round(delta), round(min_area), round(min_area + max_area), max_variation, min_diversity,
                                         round(max_evolution), round(area_threshold), min_margin, round(edge_blur_size))
        score = 0

        correct = 0
        incorrect = 0

        n_regions = 0
        mean_score = 0

        for img, particle_locations in zip(self.images, self.particle_locations):
            # Detect blobs
            blobs = self.detector.mser_detect_blobs(img)
            # Extracting stable regions
            passed_contours = self.detector.extract_regions(blobs, img, return_passed_contours=True)
            # Rectangle that best fits each region
            min_rotated_rects = [cv2.minAreaRect(blob) for blob in passed_contours]
            # Count the number of regions
            n_regions += len(min_rotated_rects)
            dims = [rect[1] for rect in min_rotated_rects]
            # Start of the Bayesian reward function
            if n_regions == 0:
                # Forcing the Bayesian optimization to stay away from parameters that result in no regions
                return -2e8

            # Iterating through all the stable rectangle regions
            for rect in min_rotated_rects:
                # Grab the center
                center = np.asarray(rect[0]).reshape(1, 2)
                # Check the distance between rectangle center and all the known rectangles
                # i.e., find the closest rectangle to the current rectangle of focus
                closest = min(np.linalg.norm(particle_locations - center, axis=1))
                # As the region that is detected moves closer to a correct particle, the higher the score will go
                mean_score -= closest
                area = rect[1][0] * rect[1][1]
                # Reject if outside minimum area
                if area < self.min_area:
                    area_diff = area - self.min_area
                # Reject if outside maximum area
                elif area > self.max_area:
                    area_diff = self.max_area - area
                else:
                    area_diff = 0

                # If the rectangle falls inside the desired min_area max_area range, then subtract nothing because area_diff = 0
                mean_score -= abs(area_diff)

            mean_score /= n_regions
            if n_regions < len(particle_locations):
                mean_score += (n_regions - len(particle_locations))*100

            for i in range(len(passed_contours)):
                contour_area = cv2.contourArea(passed_contours[i])
                bbox_area = dims[i][0] * dims[i][1]

                if bbox_area == 0:
                    continue

                ratio = contour_area / bbox_area
                if self.min_area <= bbox_area <= self.max_area and ratio >= 0.7:
                    correct += 1
                else:
                    incorrect += 1

        score = mean_score
        #print(score, correct, incorrect, n_regions)

        if score > self.best_score:
            self.best_score = score

        return score
