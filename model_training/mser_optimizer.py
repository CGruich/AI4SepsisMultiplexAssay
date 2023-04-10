import numpy as np
import os.path
import json
from object_detection import RegionDetector
import cv2
import bayes_opt
from bayes_opt import SequentialDomainReductionTransformer
SET1_1_12_IMAGE1_LABELS = np.asarray([(1095, 81  ),
(1116, 124 ),
(1152, 116 ),
(1200, 34  ),
(1238, 85  ),
(1302, 136 ),
(1340, 8   ),
(1430, 72  ),
(1564, 70  ),
(1603, 114 ),
(1825, 108 ),
(1900, 69  ),
(1969, 72  ),
(1985, 19  ),
(2017, 124 ),
(2134, 36  ),
(2262, 10  ),
(2358, 66  ),
(2451, 59  ),
(2496, 64  ),
(2533, 83  ),
(2512, 127 ),
(2437, 213 ),
(2476, 239 ),
(2425, 248 ),
(2380, 264 ),
(2481, 291 ),
(2366, 317 ),
(2329, 347 ),
(2444, 370 ),
(2189, 208 ),
(2149, 173 ),
(2154, 212 ),
(2093, 306 ),
(1949, 284 ),
(1865, 279 ),
(1804, 338 ),
(1715, 227 ),
(1641, 200 ),
(1594, 207 ),
(1487, 197 ),
(1413, 205 ),
(1302, 136 ),
(1256, 240 ),
(1386, 267 ),
(1325, 342 ),
(1502, 332 ),
(1482, 380 ),
(1539, 393 ),
(1110, 276 ),
(1046, 328 ),
(939, 287  ),
(974, 377  ),
(1076, 436 ),
(964, 443  ),
(847, 433  ),
(913, 188  ),
(988, 109  ),
(1033, 149 ),
(1094, 81  ),
(1116, 124 ),
(1153, 116 ),
(1200, 35  ),
(1238, 85  ),
(1430, 72  ),
(1563, 69  ),
(1604, 115 ),
(1121, 590 ),
(1053, 602 ),
(1068, 664 ),
(1143, 720 ),
(947, 647  ),
(865, 707  ),
(912, 739  ),
(800, 594  ),
(1314, 783 ),
(1380, 816 ),
(1206, 834 ),
(1256, 869 ),
(1312, 896 ),
(1428, 977 ),
(1445, 1026),
(1251, 983 ),
(1051, 961 ),
(1015, 1085),
(1144, 1114),
(1238, 1099),
(1273, 1151),
(1233, 1201),
(1037, 1154),
(971, 1164 ),
(797, 1110 ),
(780, 1046 ),
(809, 1017 ),
(880, 1463 ),
(634, 1529 ),
(824, 1625 ),
(994, 1763 ),
(525, 1580 ),
(464, 1579 ),
(441, 1626 )])


class MSEROptimizer(object):
    def __init__(self, normalized_images):
        self.images = normalized_images
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
        self.num_iterations = 10

    def train(self, saveDir: str = None):
        if saveDir is not None:
            self.save_filename = saveDir
        
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
        optimizer.maximize(init_points=50, n_iter=self.num_iterations)
        
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

        for img in self.images:
            blobs = self.detector.mser_detect_blobs(img)
            passed_contours = self.detector.extract_regions(blobs, img, return_passed_contours=True)
            min_rotated_rects = [cv2.minAreaRect(blob) for blob in passed_contours]
            n_regions += len(min_rotated_rects)
            dims = [rect[1] for rect in min_rotated_rects]
            if n_regions == 0:
                return -2e8

            for rect in min_rotated_rects:
                center = np.asarray(rect[0]).reshape(1, 2)
                closest = min(np.linalg.norm(SET1_1_12_IMAGE1_LABELS - center, axis=1))
                mean_score -= closest
                area = rect[1][0] * rect[1][1]
                if area < self.min_area:
                    area_diff = area - self.min_area
                elif area > self.max_area:
                    area_diff = self.max_area - area
                else:
                    area_diff = 0

                mean_score -= abs(area_diff)

            mean_score /= n_regions
            if n_regions < len(SET1_1_12_IMAGE1_LABELS):
                mean_score += (n_regions - len(SET1_1_12_IMAGE1_LABELS))*100

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
