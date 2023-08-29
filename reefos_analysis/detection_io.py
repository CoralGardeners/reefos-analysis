from collections import defaultdict
import datetime as dt
import pytz
from dataclasses import dataclass
from influxdb_client import Point


@dataclass
class Table:
    records: list


@dataclass
class Record:
    values: dict


# %%

def det_to_points(detections, labels, model_name, model_version, image_name):
    def _get_mean(dets, attr):
        # compute mean of some attribute in a list of detections
        if len(dets) == 0:
            return 0
        vals = [d[attr] for d in dets]
        return sum(vals) / len(vals)

    points = []
    # add influxdb Points for the raw detections
    for idx, d in enumerate(detections):
        point = (Point("detection")
                 .tag("model", d['model'])
                 .tag("version", d['version'])
                 .tag("class", d['class'])
                 .field("file", d['file'])
                 .field("confidence", d['conf'])
                 .field("bbox_top_left_x", d['xyxy'][0])
                 .field("bbox_top_left_y", d['xyxy'][1])
                 .field("bbox_bottom_right_x", d['xyxy'][2])
                 .field("bbox_bottom_right_y", d['xyxy'][3])
                 .field("detection_index", idx))
        points.append(point)

    # Organize detections by class
    detections_by_class = defaultdict(list)
    for d in iter(detections):
        detections_by_class[d['class']].append(d)
    # add a point for each class that was detected in the image
    for class_name, dets in detections_by_class.items():
        point = (Point("class_detection")
                 .tag("model", dets[0]['model'])
                 .tag("version", dets[0]['version'])
                 .tag("class", class_name)
                 .field("count", len(dets))
                 .field("length", _get_mean(dets, 'length'))
                 .field("depth", _get_mean(dets, 'depth'))
                 .field("file", dets[0]['file']))
        points.append(point)

    # add a Point for total number of detections in the image
    point = (Point("class_detection")
             .tag("model", model_name)
             .tag("version", model_version)
             .tag("class", "all_classes")
             .field("count", len(detections))
             .field("diversity", len(detections_by_class))
             .field("length", _get_mean(detections, 'length'))
             .field("depth", _get_mean(detections, 'depth'))
             .field("file", image_name))
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
