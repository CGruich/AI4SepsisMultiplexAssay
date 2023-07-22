import os
import cv2
import json
import numpy as np


class ParticleMarkingTool(object):
    def __init__(self, display_window_width=2400, display_window_height=1390):
        self.particle_json = {'particle_locations': []}
        self.current_file_name = ""
        self.current_image = None
        self.window_size_transform = [1, 1]
        self.display_window_size = (display_window_width, display_window_height)
        self.circles_to_render = []

    def mark_particles(self, folder_path):
        existing_files = list(os.listdir(folder_path))
        for file_name in existing_files:
            if (
                'ref' in file_name
                or 'MSER' in file_name
                or 'amp' in file_name
                or 'phase' in file_name
                or 'particle' in file_name
            ):
                continue
            img_name = os.path.join(folder_path, file_name)
            current_file_name = file_name.replace('.tiff', '_particle_locations.json')

            self.current_file_name = img_name.replace('.tiff', '_particle_locations.json')
            print('Loading image {}'.format(img_name))
            img = cv2.imread(img_name, cv2.IMREAD_ANYDEPTH)
            h, w = img.shape
            img = cv2.resize(img, self.display_window_size)
            new_h, new_w = img.shape
            self.window_size_transform = [new_w / w, new_h / h]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            self.current_image = img
            if current_file_name in existing_files:
                with open(self.current_file_name, 'r') as f:
                    self.particle_json = dict(json.load(f))
                    for coords in self.particle_json['particle_locations']:
                        x, y = coords
                        transformed_x = int(round(x * self.window_size_transform[0]))
                        transformed_y = int(round(y * self.window_size_transform[1]))
                        self.circles_to_render.append((transformed_x, transformed_y))

            cv2.namedWindow('image')
            cv2.moveWindow('image', 0, 0)
            cv2.imshow('image', self.current_image)

            cv2.setMouseCallback('image', self.mouse_callback)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print('SAVING {}\n{}'.format(self.current_file_name, self.particle_json))
            with open(self.current_file_name, 'w') as f:
                f.write(json.dumps(self.particle_json, indent=4))
            self.particle_json.clear()
            self.particle_json['particle_locations'] = []
            self.circles_to_render = []

    def mouse_callback(self, event, x, y, flags, param):
        transformed_x = int(round(x / self.window_size_transform[0]))
        transformed_y = int(round(y / self.window_size_transform[1]))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.particle_json['particle_locations'].append([transformed_x, transformed_y])
            self.circles_to_render.append((x, y))
            print('Click', transformed_x, transformed_y)

        elif event == cv2.EVENT_RBUTTONDOWN:
            try:
                coords = self.particle_json['particle_locations']
                if len(coords) == 0:
                    return

                distances = np.linalg.norm(
                    np.subtract(coords, (transformed_x, transformed_y)), axis=-1
                )

                nearest_idx = np.argmin(distances)
                nearest_dist = distances[nearest_idx]

                if nearest_dist < 100:
                    print('Removing circle at', coords[nearest_idx])
                    self.particle_json['particle_locations'].pop(nearest_idx)
                    self.circles_to_render.pop(nearest_idx)
            except:
                import traceback

                print('FAILED TO REMOVE CIRCLE')
                traceback.print_exc()

        render_image = self.current_image.copy()
        for coords in self.circles_to_render:
            cv2.circle(render_image, coords, 7, (0, 65000, 0), -1)
        cv2.imshow('image', render_image)


if __name__ == '__main__':
    #folder_path = 'C:/Users/jane/Desktop/particle_location_jsons/code 1'
    folder_path = '/home/cameron/Dropbox (University of Michigan)/DL_training/particle_markings'
    tool = ParticleMarkingTool()
    tool.mark_particles(folder_path)
