from ultralytics import YOLO
from reefos_analysis import detection_io as dio


# %%
class Detect:
    def __init__(self, device='mps', model_wts="data/model/v1.pt"):
        # load model
        self.model = YOLO(model_wts)
        self.model.fuse()
        self.model.to(device)

    def unpack_detections(self, detections, fn, model_name, model_version):
        det = []
        for idx, obj in enumerate(detections):
            box = [box.item() for box in obj[0]]
            # length, depth = get_dimensions(box, depth_map, image_width=img.width)
            det.append({
                'class': f'{self.model.names[obj[2]]}',
                'conf': f'{obj[1]:.2f}',
                #'length': length,
                #'depth': depth,
                'xyxy': box,
                'file': fn,
                'model': model_name,
                'version': model_version,
                'idx': idx
            })
        return det

    def detect_images(self, images_path="data/images"):

        results = self.model.predict(source=images_path, exist_ok=True, name='predict')

        detections = {res.path.split('/')[-1]: dio.Detections.from_ultralytics(res) for res in results}
        det = []
        for fn, dets in detections.items():
            det.extend(self.unpack_detections(dets, fn))
        return det

    def detect_image(self, img, img_name, model_name, model_version):
        results = self.model.predict(source=img, exist_ok=True, name='predict')

        detections = dio.Detections.from_ultralytics(results[0])
        det = self.unpack_detections(detections, img_name, model_name, model_version)
        return det
    
    def get_classes(self):
        return self.model.names


if __name__ == '__main__':
    detect = Detect()
    detections = detect.detect_images()
