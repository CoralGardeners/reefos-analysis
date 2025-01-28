import cv2
from ultralytics import YOLO
from reefos_analysis import detection_io as dio


# %%
class Detect:
    def __init__(self, device='mps', model_wts="data/model/v4.pt"):
        # load model
        self.model = YOLO(model_wts)
        self.model.fuse()
        self.model.to(device)

    def get_classes(self):
        return self.model.names

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

    def detect_images(self, images_path="data/images", save=False,
                      model_name=None, model_version=None, line_width=2):

        results = self.model.predict(source=images_path, exist_ok=True, name='predict',
                                     line_width=line_width, save=save, agnostic_nms=True)
        detections = {res.path.split('/')[-1]: dio.Detections.from_ultralytics(res) for res in results}
        det = []
        for fn, dets in detections.items():
            det.extend(self.unpack_detections(dets, fn, model_name, model_version))
        return det

    def detect_image(self, img, img_name, model_name, model_version, line_width=2, plot=False):
        results = self.model.predict(source=img, exist_ok=True, name='predict',
                                     line_width=line_width, agnostic_nms=True)

        detections = dio.Detections.from_ultralytics(results[0])
        det = self.unpack_detections(detections, img_name, model_name, model_version)
        if plot:
            image = results[0].plot(line_width=2, font_size=12)
            return det, image
        return det

    # detect images in folder and create video with bounding boxes
    # counts default hold frames at the start of the image for longer and then speeds up
    def detect_images_to_video(self, images_path, output_file_str, fps=30, conf=0.25, counts=[30, 15, 8, 4, 2]):
        writer = None
        results = self.model.predict(source=images_path, exist_ok=True, name='predict',
                                     save=False, agnostic_nms=True)
        for idx, result in enumerate(results):
            print(f'Frame {idx}')
            image = result.plot(line_width=2, font_size=12)
            cnt = counts[idx] if counts is not None and idx < len(counts) else 1
            if writer is None:
                writer = cv2.VideoWriter(output_file_str, cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (image.shape[1], image.shape[0]))
            while cnt > 0:
                writer.write(image)
                cnt -= 1
        writer.release()

    # detect in video and create video with bounding boxes
    def detect_video(self, video_path_str, output_path_str, device='mps', conf=0.5):
        writer = None
        results = self.model.predict(video_path_str, device=device, stream=True,
                                     conf=conf, agnostic_nms=True)
        for idx, result in enumerate(results):
            print(f'Frame {idx}')
            image = result.plot(line_width=2, font_size=12)
            if writer is None:
                writer = cv2.VideoWriter(output_path_str,
                                         cv2.VideoWriter_fourcc(*'mp4v'),
                                         30,
                                         (image.shape[1], image.shape[0]))
            writer.write(image)
        writer.release()


if __name__ == '__main__':
    detect = Detect()
    detections = detect.detect_images()
