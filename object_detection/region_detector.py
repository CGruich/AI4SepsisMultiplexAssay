import numpy as np
import cv2
from object_detection import RegionClassifier
from utils import helper_functions


class RegionDetector(object):
    def __init__(
        self,
        model_load_path: str = None,
        MSER_parameters: dict = None,
        desired_region_shape=(128, 128),
    ):
        self.bbox_threshold = 0.5
        self.desired_region_shape = desired_region_shape
        self.region_classifier = RegionClassifier(
            region_shape=(1, *self.desired_region_shape))
        if model_load_path is not None:
            self.region_classifier.load_weights(model_load_path)

        if MSER_parameters is not None:
            self.MSER_parameters = (
                round(MSER_parameters['delta']),
                round(MSER_parameters['min_area']),
                round(MSER_parameters['min_area'] +
                      MSER_parameters['max_area']),
                MSER_parameters['max_variation'],
                MSER_parameters['min_diversity'],
                round(MSER_parameters['max_evolution']),
                round(MSER_parameters['area_threshold']),
                MSER_parameters['min_margin'],
                round(MSER_parameters['edge_blur_size']),
            )
        else:
            # Determined from 13 images of Code 1 particles.
            self.MSER_parameters = (
                7.284,
                1112,
                7990,
                0.3684,
                0.8674,
                645.4,
                55.92,
                0.4834,
                162.6,
            )
            # Code_1_Ref1 -- LEGACY
            self.MSER_parameters = (
                1, 547, 1881, 1, 0.81046, 717, 794, 0.09432, 39)

    def detect_regions(self, hologram_image, reference_image, save_img_name=None):
        """
        Function to detect blobs in a hologram, construct bounding boxes around them, and use a trained classification
        model to filter out regions that do not contain objects of interest.
        :param hologram_image: Hologram to detect objects in.
        :param reference_image: Reference image with which to normalize the hologram.
        :param save_img_name: For debugging blob detection, an image with this name will be saved with MSER blobs drawn on it.
        :return: A list containing every region that an object was detected in.
        """

        grayscale_hologram = helper_functions.normalize_by_reference(
            hologram_image, reference_image
        )
        # Pass our newly normalized image to MSER for blob detection.
        detected_blobs, _ = self.mser_detect_blobs(
            grayscale_hologram,
            draw_blobs=save_img_name is not None,
            save_img_name=save_img_name,
        )

        # Get 64x64 bounding boxes around every detected blob.
        regions, _ = self.extract_regions(detected_blobs, grayscale_hologram)

        # Use a trained classifier to filter regions based on whether they contain an object of interest or not.
        positive_regions, negative_regions = self.region_classifier.classify_regions(
            regions)
        print(
            'Detected {} positive regions and {} negative regions from {} total detected regions.'.format(
                len(positive_regions), len(negative_regions), len(regions)
            )
        )

        return positive_regions, negative_regions

    def get_intensity(self, hologram_image, reference_image, save_img_name=None):
        """
        Function to visualize classified images and return intensity
        :param hologram_image: Hologram to detect objects in.
        :param reference_image: Reference image with which to normalize the hologram.
        :return: A list containing (horizontal, vertical, intensity).
                (horizontal, vertical): the center of the rectangale, where horizontal starts from the left and vertical starts from the top
        """

        grayscale_hologram = helper_functions.normalize_by_reference(
            hologram_image, reference_image
        )

        # Pass our newly normalized image to MSER for blob detection.
        detected_blobs, rects = self.mser_detect_blobs(
            grayscale_hologram,
            draw_blobs=save_img_name is not None,
            save_img_name=save_img_name,
        )

        # Get 64x64 bounding boxes around every detected blob.
        regions, picks = self.extract_regions(
            detected_blobs, grayscale_hologram)

        # Use a trained classifier to filter regions based on whether they contain an object of interest or not.
        (
            positive_regions,
            negative_regions,
            positive_idx,
        ) = self.region_classifier.classify_regions(regions, return_picks=True)

        positive_idx = [picks[i] for i in positive_idx]
        positive_recs = [rects[i] for i in positive_idx]

        vis = grayscale_hologram.copy()
        # Convert the copy to 8-bit.
        vis = np.divide(vis, 256)
        vis = vis.astype('uint8')
        # Convert the copy to BGR format.
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        for r in positive_recs:
            cv2.drawContours(vis, [r], 0, (0, 0, 65000), 1)
        if save_img_name is None:
            save_img_name = 'data/test/MSER Hulls.png'
        cv2.imwrite(save_img_name, vis)

        print(
            'Detected {} positive regions and {} negative regions from {} total detected regions.'.format(
                len(positive_regions), len(negative_regions), len(regions)
            )
        )

        intensity = []
        for rec in positive_recs:
            # make a mask
            mask = np.zeros_like(grayscale_hologram)
            mask = cv2.fillPoly(mask, pts=[rec], color=(1)).astype(bool)

            intensity.append(
                [
                    np.average(rec, axis=0)[0],
                    np.average(rec, axis=0)[1],
                    np.sum(grayscale_hologram[mask]) / np.sum(mask),
                ]
            )

            # rotated_rec = cv2.rotatedRect(cv2.Point2f(rec[0]), cv2.Point2f(rec[1]), cv2.Point2f(rec[2]))
            # intensity.append()

        return intensity

    def mser_detect_blobs(self, grayscale_hologram, draw_blobs=False, save_img_name=None):
        """
        Function to detect objects in a hologram with the MSER blob detection algorithm.
        :param grayscale_hologram: Normalized 16-bit grayscale hologram.
        :return: Array containing the convex hull surrounding every detected blob.
        """

        # Unpack the MSER parameters in our constructor.
        (
            delta,
            min_area,
            max_area,
            max_variation,
            min_diversity,
            max_evolution,
            area_threshold,
            min_margin,
            edge_blur_size,
        ) = self.MSER_parameters

        # Build the MSER object.
        mser = cv2.MSER_create(
            delta=delta,
            min_area=min_area,
            max_area=max_area,
            max_variation=max_variation,
            min_diversity=min_diversity,
            max_evolution=max_evolution,
            area_threshold=area_threshold,
            min_margin=min_margin,
            edge_blur_size=edge_blur_size,
        )

        # MSER wants the image to have a bit-depth of 8, so we'll convert the 16-bit grayscale image we made earlier
        # to an 8-bit grayscale image here.
        if grayscale_hologram.dtype == 'uint16':
            # 2^16 / 2^8 = 2^8
            img = np.divide(grayscale_hologram.astype(np.float32), 256)
            img = img.astype('uint8')
        else:
            img = grayscale_hologram

        # User MSER to detect regions in the image.
        regions, _ = mser.detectRegions(img)

        # Extract convex hulls from the resulting regions.
        blobs = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        min_rotated_rects = [cv2.minAreaRect(blob) for blob in blobs]
        dims = [rect[1] for rect in min_rotated_rects]

        blobs_to_keep = []
        for i in range(len(dims)):
            w, h = dims[i]
            if 0.8 < w / h < 1.2:
                blobs_to_keep.append(blobs[i])

        blobs = blobs_to_keep
        # Optionally draw the hulls on a display image.
        if draw_blobs:
            # Copy of input hologram for display purposes.
            vis = grayscale_hologram.copy().astype(np.float32)

            # Convert the copy to 8-bit.
            vis = np.divide(vis, 256)
            vis = vis.astype('uint8')

            # Convert the copy to BGR format.
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

            cv2.polylines(vis, blobs, 1, (0, 65000, 0))

            passed_contours = self.extract_regions(
                blobs, img, return_passed_contours=True)
            min_rotated_rects = [cv2.minAreaRect(
                blob) for blob in passed_contours]

            rects = [cv2.boxPoints(rect).astype(np.int32)
                     for rect in min_rotated_rects]
            print('detected', len(rects), 'blobs')
            for r in rects:
                cv2.drawContours(vis, [r], 0, (0, 0, 65000), 2)

            cv2.namedWindow('vis')
            cv2.imshow('vis', cv2.resize(vis, (1920, 1080)))
            cv2.moveWindow('vis', 0, 0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if save_img_name is None:
                save_img_name = 'data/test/MSER Hulls.png'
            cv2.imwrite(save_img_name, vis)

            return blobs, rects
        return blobs

    def extract_regions(self, blobs, grayscale_hologram, return_passed_contours=False):
        """
        Function to turn blobs into regions on a hologram.
        :param blobs: Contours detected by the MSER algorithm.
        :param grayscale_hologram: Hologram to extract regions from.
        :return: Array containing all the regions detected by MSER.
        """

        bboxes = []

        # For all contours detected by MSER.
        for cnt in blobs:
            # Find and save the minimally sized boundary box around each contour.
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append([x, y, x + w, y + h])

        bboxes = np.asarray(bboxes)

        # Remove all bounding boxes that overlap another box by 20% or more.
        if return_passed_contours:
            bboxes, idxs = helper_functions.non_max_suppression_fast(
                bboxes, self.bbox_threshold, return_picks=True
            )
            blobList = [blobs[idx] for idx in idxs]
            return blobList

        bboxes, idxs = helper_functions.non_max_suppression_fast(
            bboxes, self.bbox_threshold, return_picks=True
        )

        # Expand all remaining bounding boxes to desired shape.
        regions = []
        idx = 0
        idxs = []
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2, cx, cy = helper_functions.expand_bbox(
                box, self.desired_region_shape, grayscale_hologram.shape
            )
            region = np.asarray(grayscale_hologram[y1:y2, x1:x2])
            idx += 1

            # Because some contours might be at the edge of the image, we might get regions that can't be expanded.
            # When that happens, we'll just discard them.
            if region.shape != self.desired_region_shape:
                continue

            # Ensure the color channel is part of the shape of the expanded region.
            regions.append(region.reshape((1, *self.desired_region_shape)))
            idxs.append(i)

        # Return all detected regions.
        return regions, idxs
