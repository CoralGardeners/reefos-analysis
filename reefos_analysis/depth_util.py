import numpy as np


def get_bbox_dimensions(box, depth_map, horizontal_fov=110, image_width=1920):
    # Correct for underwater
    horizontal_fov = horizontal_fov / 1.33
    depth_map = depth_map * 1.33

    x1, y1, x2, y2 = map(int, box)
    depth = np.mean(depth_map[y1:y2, x1:x2][np.isfinite(depth_map[y1:y2, x1:x2])])
    pixel_width = x2 - x1
    angle = (horizontal_fov * pixel_width) / (2 * image_width)
    length = 2 * depth * np.tan(np.radians(angle))

    length = float(f'{length:.4g}')
    depth = float(f'{depth:.4g}')

    return length, depth
