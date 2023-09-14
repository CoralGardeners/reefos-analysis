import os
from collections import defaultdict
import datetime as dt
import pytz
import numpy as np
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterator, Optional, Tuple


import influxdb_client
from influxdb_client import Point, Bucket, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS


@dataclass
class Table:
    records: list


@dataclass
class Record:
    values: dict


# %%

def det_to_points(detections, image_name, model_name, model_version):
    def _get_mean(dets, attr):
        # compute mean of some attribute in a list of detections
        if len(dets) == 0:
            return 0
        vals = [d[attr] for d in dets]
        return sum(vals) / len(vals)

    points = []
    timestamp = get_filename_time(image_name)
    # add influxdb Points for the raw detections
    for idx, d in enumerate(detections):
        point = (Point("raw_detection")
                 .tag("model", model_name)
                 .tag("version", model_version)
                 .tag("class", d['class'])
                 .field("file", image_name)
                 .field("confidence", d['conf'])
                 .field("bbox_top_left_x", d['xyxy'][0])
                 .field("bbox_top_left_y", d['xyxy'][1])
                 .field("bbox_bottom_right_x", d['xyxy'][2])
                 .field("bbox_bottom_right_y", d['xyxy'][3])
                 .field("detection_index", idx)
                 .time(timestamp)
                 )
        points.append(point)

    # Organize detections by class
    detections_by_class = defaultdict(list)
    for d in iter(detections):
        detections_by_class[d['class']].append(d)
    # add a point for each class that was detected in the image
    for class_name, dets in detections_by_class.items():
        point = (Point("class_detection")
                 .tag("model", model_name)
                 .tag("version", model_version)
                 .tag("class", class_name)
                 .field("count", len(dets))
                 .field("length", _get_mean(dets, 'length'))
                 .field("depth", _get_mean(dets, 'depth'))
                 .field("file", image_name)
                 .time(timestamp)
                 )
        points.append(point)

    # add a Point for total number of detections in the image
    point = (Point("class_detection")
             .tag("model", model_name)
             .tag("version", model_version)
             .tag("class", "all_classes")
             .field("file", image_name)
             .field("count", len(detections))
             .field("diversity", len(detections_by_class))
             .field("length", _get_mean(detections, 'length'))
             .field("depth", _get_mean(detections, 'depth'))
             .time(timestamp)
             )
    points.append(point)
    return points


def get_filename_time(fn):
    dt_notz = dt.datetime.strptime(fn.split('.')[0], "%m_%d_%Y_%H_%M_%S")
    tahiti_tz = pytz.timezone('Pacific/Tahiti')
    dt_tahiti = tahiti_tz.localize(dt_notz)
    return dt_tahiti



def mock_influx_query_results(detections):
    results = []
    for idx, det in enumerate(detections):
        tm = get_filename_time(det['file'])
        field_val = {'bbox_top_left_x': det['xyxy'][0],
                     'bbox_top_left_y': det['xyxy'][1],
                     'bbox_bottom_right_x': det['xyxy'][2],
                     'bbox_bottom_right_y': det['xyxy'][3],
                     'confidence': det['conf']
                     }
        tag_val = {'class': det['class'],
                   'file': det['file'],
                   'result': '_result',
                   'table': idx,
                   }
        for k, v in field_val.items():
            res = tag_val.copy()
            res['_field'] = k
            res['_value'] = v
            res['_time'] = tm
            results.append(Record(res))
    return [Table(records=results)]


# %%
influx_write_api = None


def setup_influx(influxdb_url, influxdb_token, influxdb_org):
    # Set up Influx
    global influx_write_api
    if influx_write_api is None:
        write_client = influxdb_client.InfluxDBClient(url=influxdb_url, token=influxdb_token, org=influxdb_org)
        influx_write_api = write_client.write_api(write_options=SYNCHRONOUS)
    return influx_write_api


def update_influx(det, image_name, bucket_name, model_name, model_version, env):
    write_api = setup_influx(env.influxdb_url, env.influxdb_token, env.influxdb_org)

    points = det_to_points(det, image_name, model_name, model_version)
    write_api.write(bucket=bucket_name, record=points, write_precision=WritePrecision.S)

# %%
@dataclass
class Detections:
    """
    Data class containing information about object detections
    Attributes:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        confidence (Optional[np.ndarray]): An array of shape
            `(n,)` containing the confidence scores of the detections.
        class_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the class ids of the detections.
        tracker_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the tracker ids of the detections.
    """

    xyxy: np.ndarray
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None

    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)

    def __iter__(self):
        """
        Iterates over the Detections object and yield a tuple of
        `(xyxy, confidence, class_id, tracker_id)` for each detection.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None,
            )

    @classmethod
    def from_ultralytics(cls, ultralytics_results):
        """
        Creates a Detections instance from a
            [YOLOv8](https://github.com/ultralytics/ultralytics) inference result.

        Args:
            ultralytics_results (ultralytics.yolo.engine.results.Results):
                The output Results instance from YOLOv8

        Returns:
            Detections: A new Detections object.
        """

        return cls(
            xyxy=ultralytics_results.boxes.xyxy.cpu().numpy(),
            confidence=ultralytics_results.boxes.conf.cpu().numpy(),
            class_id=ultralytics_results.boxes.cls.cpu().numpy().astype(int),
            tracker_id=(ultralytics_results.boxes.id.int().cpu().numpy()
                        if ultralytics_results.boxes.id is not None
                        else None),
        )
