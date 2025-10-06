"""
Object Tracking, Speed, Direction, and Lat/Lon Extraction from RGB Camera Feed
- Polygon ROI selection (user clicks points, right-click to finish)
- ROI-based speed measurement: record entry + exit frame; speed = (avg_object_length) / time
- Ray->ground intersection primary for lat/lon;
- Robust config defaults and safe Excel saving
Author: adapted for user
Date: [Date]
"""

# Standard libs
import os
import time
import math
import json
import tempfile
from collections import defaultdict

# Third-party libs
import cv2
import numpy as np
import pandas as pd
import torch

# YOLO
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ----------------------------
# Utility / geometry functions
# ----------------------------
def get_device():
    """Return 'cuda' if GPU available else 'cpu'."""
    if torch.cuda.is_available():
        try:
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        return 'cuda'
    return 'cpu'

def load_config(config_path='config.json'):
    """
    Load camera and system configuration from JSON file.
    
    This function reads the configuration file containing camera parameters,
    detection settings, tracking parameters, and output preferences. The config
    file includes camera position (lat/lon/alt), orientation, field of view,
    detection thresholds, and display options.
    
    Args:
        config_path (str): Path to the JSON configuration file. Defaults to 'config.json'.
        
    Returns:
        dict: Configuration dictionary containing all system parameters, or None if loading fails.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    try:
        # Open and read the JSON configuration file
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Using default values.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing configuration file: {e}")
        return None
    
# ----------------------------
# Location and geometry functions
# ----------------------------

def rotation_from_rpy(roll_deg, pitch_deg, yaw_deg=0.0):
    """Return rotation matrix R = Rz(yaw)*Ry(pitch)*Rx(roll). Angles in degrees."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(r), -math.sin(r)],
                   [0, math.sin(r), math.cos(r)]], dtype=float)
    Ry = np.array([[math.cos(p), 0, math.sin(p)],
                   [0, 1, 0],
                   [-math.sin(p), 0, math.cos(p)]], dtype=float)
    Rz = np.array([[math.cos(y), -math.sin(y), 0],
                   [math.sin(y), math.cos(y), 0],
                   [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx

def enu_to_latlon(d_east, d_north, camera_lat):
    """Convert meters east/north to lat/lon deltas (approx)."""
    d_lat = d_north / 111320.0
    # avoid division by zero
    denom = 111320.0 * math.cos(math.radians(camera_lat))
    denom = denom if abs(denom) > 1e-9 else 111320.0
    d_lon = d_east / denom
    return d_lat, d_lon

def bearing_from_gps(lat1, lon1, lat2, lon2):
    """Return compass bearing (0=N, 90=E) from point1 to point2."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    angle = math.degrees(math.atan2(x, y))
    return (angle + 360.0) % 360.0

def haversine_meters(lat1, lon1, lat2, lon2):
    """Haversine distance in meters."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(min(1.0, math.sqrt(a)))

def get_camera_intrinsics(config, image_width, image_height):
    """Compute intrinsics from FOV and image size (fx,fy,cx,cy)."""
    fov_h = config['camera']['field_of_view'].get('horizontal_degrees', 90.0)
    fov_v = config['camera']['field_of_view'].get('vertical_degrees', 60.0)
    fx = (image_width / 2.0) / math.tan(math.radians(fov_h / 2.0))
    fy = (image_height / 2.0) / math.tan(math.radians(fov_v / 2.0))
    cx = image_width / 2.0
    cy = image_height / 2.0
    return {'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy)}

def pixel_depth_to_world(x, y, depth, intrinsics):
    """Pixel->camera 3D coordinate using pinhole model."""
    fx = intrinsics['fx']; fy = intrinsics['fy']; cx = intrinsics['cx']; cy = intrinsics['cy']
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z], dtype=float)

# Ray helpers
def cam_ray_from_pixel(cx_pixel, cy_pixel, intrinsics):
    """Return normalized ray in camera coords for pixel."""
    fx = intrinsics['fx']; fy = intrinsics['fy']; cx = intrinsics['cx']; cy = intrinsics['cy']
    x = (cx_pixel - cx) / fx
    y = (cy_pixel - cy) / fy
    z = 1.0
    v = np.array([x, y, z], dtype=float)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def cam_ray_to_enu_direction(ray_cam, camera_roll_deg, camera_pitch_deg, camera_facing_deg, calibration_yaw_deg):
    """
    Transform a normalized camera-ray to ENU direction vector.
    Uses roll/pitch to orient camera, then cam->enu convention, then facing + calibration yaw.
    """
    # Camera coordinates -> apply roll & pitch
    R_cam = rotation_from_rpy(camera_roll_deg, camera_pitch_deg, 0.0)
    ray_after_camera = R_cam @ ray_cam
    # Camera->ENU mapping: X_cam -> east, Y_cam (down) -> -north, Z_cam -> up
    cam_to_enu = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)
    enu_intermediate = cam_to_enu @ ray_after_camera
    # Apply facing + calibration yaw (rotate around up axis)
    R_facing = rotation_from_rpy(0.0, 0.0, camera_facing_deg + calibration_yaw_deg)
    ray_enu = R_facing @ enu_intermediate
    ray_enu = ray_enu / (np.linalg.norm(ray_enu) + 1e-12)
    return ray_enu

def camera_to_world_from_enu(enu_point, camera_lat, camera_lon, camera_alt):
    """Convert ENU (absolute) to global lat/lon/alt. ENU given relative to camera origin at camera_alt."""
    if enu_point is None:
        return None, None, None
    d_east, d_north, d_up = float(enu_point[0]), float(enu_point[1]), float(enu_point[2])
    d_lat, d_lon = enu_to_latlon(d_east, d_north, camera_lat)
    return (camera_lat + d_lat, camera_lon + d_lon, camera_alt + d_up)

# ----------------------------
# Detection processing pipeline
# ----------------------------
def process_detections(results, track_classes, min_bbox_area,
                       camera_intrinsics, camera_cfg):
    """
    Process YOLO detections and compute 3D positions using:
      1. Ray→ground intersection
      2. Size-based depth estimation (if object_real_sizes provided)
      3. Constant fallback depth
    """

    detections = []
    camera_lat = camera_cfg['position']['latitude']
    camera_lon = camera_cfg['position']['longitude']
    camera_alt = camera_cfg['position'].get('altitude', 0.0)
    camera_facing_deg = camera_cfg['orientation'].get('facing_degrees', 0.0)
    camera_pitch_deg = camera_cfg['orientation'].get('pitch_degrees', 0.0)
    camera_roll_deg = camera_cfg['orientation'].get('roll_degrees', 0.0)
    calibration_yaw_deg = camera_cfg['orientation'].get('calibration_yaw_degrees', 0.0)

    # Defaults (no depth section required)
    default_depth_m = 30.0
    min_depth_m = 0.1
    object_real_sizes = camera_cfg.get("object_real_sizes", {})

    for box in results.boxes:
        cls = int(box.cls[0])
        oid = int(box.id[0]) if getattr(box, "id", None) is not None else -1
        if cls not in track_classes:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w_box, h_box = max(0, x2 - x1), max(0, y2 - y1)
        if w_box * h_box < min_bbox_area:
            continue
        cx = int((x1 + x2) / 2)
        cy = int(y2)

        # Build ray from pixel
        ray_cam = cam_ray_from_pixel(cx, cy, camera_intrinsics)
        ray_enu = cam_ray_to_enu_direction(ray_cam,
                                           camera_roll_deg, camera_pitch_deg,
                                           camera_facing_deg, calibration_yaw_deg)

        cam_pos_enu = np.array([0.0, 0.0, camera_alt], dtype=float)

        depth_used = None
        xyz_cam = None
        enu_point = None

        # 1) Ray->ground intersection
        if ray_enu[2] < -1e-6:
            t = -cam_pos_enu[2] / ray_enu[2]
            if t > 0:
                enu_point = cam_pos_enu + t * ray_enu
                distance = np.linalg.norm(enu_point - cam_pos_enu)
                xyz_cam = ray_cam * distance
                depth_used = float(xyz_cam[2])

        # 2) Size-based fallback
        if xyz_cam is None and str(cls) in object_real_sizes:
            try:
                H_real = float(object_real_sizes[str(cls)])  # meters
                fy = camera_intrinsics['fy']
                depth_est = (fy * H_real) / (h_box + 1e-9)
                if depth_est > min_depth_m:
                    depth_used = float(depth_est)
                    X_cam = (cx - camera_intrinsics['cx']) * depth_used / camera_intrinsics['fx']
                    Y_cam = (cy - camera_intrinsics['cy']) * depth_used / camera_intrinsics['fy']
                    Z_cam = depth_used
                    xyz_cam = np.array([X_cam, Y_cam, Z_cam], dtype=float)
                    R_cam = rotation_from_rpy(camera_roll_deg, camera_pitch_deg, 0.0)
                    cam_to_enu = np.array([[1, 0, 0],
                                           [0, -1, 0],
                                           [0, 0, 1]], dtype=float)
                    enu_xyz = cam_to_enu @ (R_cam @ xyz_cam)
                    R_facing = rotation_from_rpy(0.0, 0.0,
                                                 camera_facing_deg + calibration_yaw_deg)
                    enu_point = R_facing @ enu_xyz + np.array([0.0, 0.0, camera_alt], dtype=float)
            except Exception as e:
                print("Size-based depth estimation error:", e)

        # 3) Final fallback constant depth
        if xyz_cam is None:
            depth_used = default_depth_m
            X_cam = (cx - camera_intrinsics['cx']) * depth_used / camera_intrinsics['fx']
            Y_cam = (cy - camera_intrinsics['cy']) * depth_used / camera_intrinsics['fy']
            Z_cam = depth_used
            xyz_cam = np.array([X_cam, Y_cam, Z_cam], dtype=float)
            R_cam = rotation_from_rpy(camera_roll_deg, camera_pitch_deg, 0.0)
            cam_to_enu = np.array([[1, 0, 0],
                                   [0, -1, 0],
                                   [0, 0, 1]], dtype=float)
            enu_xyz = cam_to_enu @ (R_cam @ xyz_cam)
            R_facing = rotation_from_rpy(0.0, 0.0,
                                         camera_facing_deg + calibration_yaw_deg)
            enu_point = R_facing @ enu_xyz + np.array([0.0, 0.0, camera_alt], dtype=float)

        detections.append({
            "id": oid,
            "xyz": xyz_cam,
            "class": cls,
            "cx": cx,
            "cy": cy,
            "bbox": (x1, y1, x2, y2),
            "conf": conf,
            "depth_used_m": float(depth_used) if depth_used is not None else None,
            "enu_point": enu_point
        })

    return detections

# ----------------------------
# Update & annotation functions
# ----------------------------
def calculate_speed(trajectory, fps, window=5):
    """Return speed in km/h using last `window` points in trajectory list [(frame, xyz), ...]."""
    if len(trajectory) < 2 or fps is None or fps <= 0:
        return 0.0
    pts = trajectory[-window:] if len(trajectory) >= window else trajectory
    (f0, p0), (f1, p1) = pts[0], pts[-1]
    dist = np.linalg.norm(np.array(p1) - np.array(p0))
    frame_diff = (f1 - f0)
    if frame_diff <= 0:
        return 0.0
    time_s = frame_diff / float(fps)
    speed_mps = dist / time_s
    return speed_mps * 3.6

def update_object_data(det, object_data, internal_id, camera_lat, camera_lon, camera_alt,
                       camera_facing_deg, camera_roll_deg, camera_pitch_deg, calibration_yaw_deg,
                       fps, speed_window, frame_idx, class_names):
    """Update histories and compute lat/lon/speed/direction. Returns a row dict for framewise export."""
    xyz = det['xyz']
    cls_id = det['class']
    bbox = det['bbox']
    conf = det['conf']
    cx = det['cx']; cy = det['cy']
    enu_point = det.get('enu_point', None)

    # convert enu to lat/lon
    lat, lon, alt = (None, None, None)
    if enu_point is not None:
        lat, lon, alt = camera_to_world_from_enu(enu_point, camera_lat, camera_lon + 0.00014, camera_alt)

    # compute speed from camera-frame xyz history
    speed = calculate_speed(object_data[internal_id]['xyz_history'], fps, window=speed_window)

    # compute direction using last lat/lon if available
    direction = ''
    if len(object_data[internal_id]['lat']) >= 1 and lat is not None:
        prev_lat = object_data[internal_id]['lat'][-1]
        prev_lon = object_data[internal_id]['lon'][-1] if len(object_data[internal_id]['lon']) >= 1 else None
        if prev_lat is not None and prev_lon is not None and lat is not None and lon is not None:
            direction = bearing_from_gps(prev_lat, prev_lon, lat, lon)

    # append to history
    object_data[internal_id]['lat'].append(lat)
    object_data[internal_id]['lon'].append(lon)
    object_data[internal_id]['speed'].append(speed)
    object_data[internal_id]['direction'].append(direction)
    object_data[internal_id]['xyz_history'].append((frame_idx, np.array(xyz if xyz is not None else [0.0, 0.0, 0.0], dtype=float)))

    row = {
        'frame': frame_idx,
        'id': internal_id,
        'class_id': cls_id,
        'class_name': class_names.get(cls_id, 'UNK'),
        'bbox_x1': bbox[0],
        'bbox_y1': bbox[1],
        'bbox_x2': bbox[2],
        'bbox_y2': bbox[3],
        'cx': cx,
        'cy': cy,
        'depth_m': float(det.get('depth_used_m', np.nan)) if det.get('depth_used_m') is not None else np.nan,
        'lat': lat,
        'lon': lon,
        'alt': alt,
        'speed_kmh': speed,
        'direction_deg': direction,
        'conf': conf
    }
    return row

def draw_annotations(frame, det, internal_id, cls_id, speed, direction, lat, lon, camera_intrinsics,
                     show_id, show_class, show_direction, show_speed, show_latlon, class_names):
    """Draw label and marker at projected point from camera-frame xyz."""
    xyz = det.get('xyz', None)
    if xyz is None:
        return
    fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
    cx_i, cy_i = camera_intrinsics['cx'], camera_intrinsics['cy']
    if abs(xyz[2]) < 1e-6:
        return
    px = int(xyz[0] * fx / xyz[2] + cx_i)
    py = int(xyz[1] * fy / xyz[2] + cy_i)
    label_parts = []
    if show_id:
        label_parts.append(f"ID:{internal_id}")
    if show_class:
        label_parts.append(class_names.get(cls_id, 'UNK'))
    if show_direction and direction != '':
        label_parts.append(f"{direction:.1f}°")
    if show_speed:
        label_parts.append(f"{speed:.1f} km/h")
    if show_latlon and (lat is not None and lon is not None):
        label_parts.append(f"{lat:.6f},{lon:.6f}")
    label = " | ".join(label_parts)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    tl = (max(0, px), max(0, py - th - baseline - 4))
    br = (min(frame.shape[1], px + tw + 4), min(frame.shape[0], py + baseline + 4))
    cv2.rectangle(frame, tl, br, (0,0,0), thickness=-1)
    cv2.putText(frame, label, (px, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.circle(frame, (px, py), 4, (0,255,0), -1)

def safe_save_excel(df_obj, base_filename):
    """Save DataFrame to Excel. If PermissionError, append timestamp to name."""
    try:
        df_obj.to_excel(base_filename, index=False)
        print(f"Excel file saved as {base_filename}")
        return base_filename
    except PermissionError:
        ts = int(time.time())
        alt = base_filename.replace('.xlsx', f'_{ts}.xlsx')
        df_obj.to_excel(alt, index=False)
        print(f"Could not write to {base_filename}. Saved as {alt}")
        return alt

def create_summary_dataframe(object_data):
    """Return DataFrame summarizing tracked objects (avg lat/lon/speed/direction)."""
    rows = []
    for oid, data in object_data.items():
        # ensure we have at least some lat/lon entries
        valid_lats = [v for v in data['lat'] if v is not None]
        valid_lons = [v for v in data['lon'] if v is not None]
        if valid_lats and valid_lons:
            avg_lat = sum(valid_lats) / len(valid_lats)
            avg_lon = sum(valid_lons) / len(valid_lons)
        else:
            avg_lat, avg_lon = None, None
        avg_speed = (sum([s for s in data['speed']]) / len(data['speed'])) if data['speed'] else 0.0
        dir_values = [d for d in data['direction'] if isinstance(d, (int, float))]
        if dir_values:
            radians = [math.radians(float(d)) for d in dir_values]
            mean_angle = math.degrees(math.atan2(np.mean([math.sin(a) for a in radians]), np.mean([math.cos(a) for a in radians]))) % 360.0
            avg_dir = mean_angle
        else:
            avg_dir = ''
        rows.append({'id': oid, 'lat': avg_lat, 'lon': avg_lon, 'speed': avg_speed, 'direction': avg_dir})
    return pd.DataFrame(rows, columns=['id','lat','lon','speed','direction'])

# ----------------------------
# ROI polygon selection helpers (GUI)
# ----------------------------
ROI_POINTS = []
ROI_DRAWING = False

def roi_mouse_callback(event, x, y, flags, param):
    """Mouse callback to collect polygon points. Left click adds, right click finishes."""
    global ROI_POINTS, ROI_DRAWING
    if event == cv2.EVENT_LBUTTONDOWN:
        ROI_POINTS.append((int(x), int(y)))
    elif event == cv2.EVENT_RBUTTONDOWN:
        ROI_DRAWING = False
        # user finished polygon selection

# ----------------------------
# Main pipeline
# ----------------------------
def main():
    # load config
    config = load_config('config.json')
    device = get_device()
    print(f"Device: {device}")

    # video path
    video_path = config['paths'].get('video_path')
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video info: {{'width':{width}, 'height':{height}, 'fps':{fps}}}")

    # camera values
    camera_cfg = config['camera']
    camera_lat = camera_cfg['position'].get('latitude', 0.0)
    camera_lon = camera_cfg['position'].get('longitude', 0.0)
    camera_alt = camera_cfg['position'].get('altitude', 0.0)
    camera_facing_deg = camera_cfg['orientation'].get('facing_degrees', 0.0)
    camera_pitch_deg = camera_cfg['orientation'].get('pitch_degrees', 0.0)
    camera_roll_deg = camera_cfg['orientation'].get('roll_degrees', 0.0)
    calibration_yaw_deg = camera_cfg['orientation'].get('calibration_yaw_degrees', 0.0)

    camera_intrinsics = get_camera_intrinsics(config, width, height)
    print("Camera intrinsics:", camera_intrinsics)

        

    # Load YOLO
    model_name = config['paths'].get('model_name', 'rtdetr-l.pt')
    if YOLO is None:
        print("Ultralytics YOLO not available. Install ultralytics package.")
        return
    model = YOLO(model_name)
    if device == 'cuda':
        try:
            model.to(device)
            print("YOLO moved to GPU")
        except Exception:
            pass

    # detection/tracking params
    track_classes = list(config['detection'].get('track_classes', []))
    class_names = {int(k): v for k, v in config['detection'].get('class_names', {}).items()}
    detection_conf_thresh = float(config['detection'].get('confidence_threshold', 0.25))
    nms_iou_thresh = float(config['detection'].get('nms_iou_threshold', 0.45))
    min_bbox_area = int(config['detection'].get('min_bbox_area', 200))

    tracker_type = config['tracking'].get('tracker_type', None)
    with_reid = bool(config['tracking'].get('with_reid', False))
    speed_window = int(config['tracking'].get('speed_calculation_window', 5))

    show_direction = bool(config['display'].get('show_direction', True))
    show_speed = bool(config['display'].get('show_speed', True))
    show_latlon = bool(config['display'].get('show_latlon', True))
    show_id = bool(config['display'].get('show_id', True))
    show_class = bool(config['display'].get('show_class', True))

    save_video_flag = bool(config['output'].get('save_video', False))
    output_video_path = config['paths'].get('output_video_path', 'output_annotated.mp4')
    save_framewise_flag = bool(config['output'].get('save_framewise', True))
    framewise_excel = config['paths'].get('framewise_excel', 'framewise.xlsx')
    summary_excel = config['paths'].get('summary_excel', 'summary.xlsx')

    # ROI polygon: either from config or user draws
    global ROI_POINTS, ROI_DRAWING
    ROI_POINTS = []
    ROI_DRAWING = True
    roi_polygon = None

    if 'roi_polygon' in config:
        # expect list of [ [x,y], [x,y], ... ]
        roi_polygon = np.array(config['roi_polygon'], dtype=np.int32)
        ROI_DRAWING = False
    else:
        # ask user to draw polygon on first frame
        ret0, frame0 = cap.read()
        if not ret0:
            print("Cannot read first frame for ROI selection.")
            return
        print("Draw polygon ROI: left-click to add vertices, right-click to finish.")
        ROI_POINTS = []
        ROI_DRAWING = True
        cv2.namedWindow("Define ROI")
        cv2.setMouseCallback("Define ROI", roi_mouse_callback)
        while ROI_DRAWING:
            tmp = frame0.copy()
            if len(ROI_POINTS) > 0:
                for p in ROI_POINTS:
                    cv2.circle(tmp, p, 3, (0,255,0), -1)
            if len(ROI_POINTS) > 1:
                cv2.polylines(tmp, [np.array(ROI_POINTS, np.int32)], isClosed=False, color=(0,255,0), thickness=2)
            cv2.imshow("Define ROI", tmp)
            k = cv2.waitKey(10) & 0xFF
            # right-click will set ROI_DRAWING False via callback; Esc also cancels
            if k == 27:  # Esc
                ROI_DRAWING = False
                break
        cv2.destroyWindow("Define ROI")
        if len(ROI_POINTS) >= 3:
            roi_polygon = np.array(ROI_POINTS, dtype=np.int32)
        else:
            print("ROI polygon not defined or insufficient points. Exiting.")
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # storage
    frame_idx = 0
    framewise_rows = []
    object_data = defaultdict(lambda: {'lat': [], 'lon': [], 'speed': [], 'direction': [], 'xyz_history': [], 'roi_entry_frame': None, 'roi_exit_frame': None})
    video_writer = None

    # process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # write video writer
        if video_writer is None and save_video_flag:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

        # draw roi polygon on frame for visualization
        try:
            cv2.polylines(frame, [roi_polygon], isClosed=True, color=(0,255,0), thickness=2)
        except Exception:
            pass

        # prepare tracker argument for Ultralytics
        tracker_arg = tracker_type
        temp_yaml_path = None
        if isinstance(tracker_type, str) and 'botsort' in tracker_type.lower() and with_reid:
            botsort_yaml = (
                'tracker_type: botsort\n'
                'track_high_thresh: 0.6\n'
                'track_low_thresh: 0.1\n'
                'new_track_thresh: 0.55\n'
                'track_buffer: 40\n'
                'match_thresh: 0.9\n'
                'proximity_thresh: 0.5\n'
                'appearance_thresh: 0.25\n'
                'fuse_score: true\n'
                'mot20: false\n'
                f"with_reid: {str(with_reid)}\n"
                'model: auto\n'
                'gmc_method: none\n'
            )
            tmp = tempfile.NamedTemporaryFile(prefix='botsort_reid_', suffix='.yaml', delete=False)
            tmp.write(botsort_yaml.encode('utf-8')); tmp.flush(); tmp.close()
            tracker_arg = tmp.name
            temp_yaml_path = tmp.name

        # run YOLO tracking / detection
        results = model.track(
            frame,
            conf=detection_conf_thresh,
            iou=nms_iou_thresh,
            persist=True,
            tracker=tracker_arg
        )[0]
        

        # process detections -> xyz, enu etc
        detections = process_detections(results, track_classes, min_bbox_area, camera_intrinsics, camera_cfg)

        # iterate dets and apply polygon mask and ROI entry/exit logic
        for det in detections:
            oid = det['id']
            if oid is None or oid == -1:
                continue
            internal_id = int(oid)

            # check center inside polygon (cv2.pointPolygonTest)
            pt = (int(det['cx']), int(det['cy']))
            inside = cv2.pointPolygonTest(roi_polygon, pt, False)
            if inside < 0:
                # If object previously inside and now outside, mark exit frame if not set
                # but do not process detections outside ROI
                if object_data[internal_id]['roi_entry_frame'] is not None and object_data[internal_id]['roi_exit_frame'] is None:
                    object_data[internal_id]['roi_exit_frame'] = frame_idx
                    # compute ROI speed if possible
                    entry = object_data[internal_id]['roi_entry_frame']
                    exitf = object_data[internal_id]['roi_exit_frame']
                    if entry is not None and exitf is not None and exitf > entry:
                        elapsed = exitf - entry
                        time_sec = elapsed / float(fps) if fps > 0 else None
                        # distance approximation uses user-provided avg length per class_name
                        cls_name = class_names.get(det['class'], str(det['class']))
                        avg_length = None
                        # config object_real_sizes might map class IDs or names -> lengths
                        if 'object_real_sizes' in config:
                            ors = config['object_real_sizes']
                            if str(det['class']) in ors:
                                avg_length = float(ors[str(det['class'])])
                            elif cls_name in ors:
                                avg_length = float(ors[cls_name])
                        if avg_length is None:
                            avg_length = float(config.get('default_object_length_m', 4.5))
        
                continue  # skip further handling of detection since outside polygon

            # If inside polygon: update object history & ROI entry frame if not set
            if object_data[internal_id]['roi_entry_frame'] is None:
                object_data[internal_id]['roi_entry_frame'] = frame_idx

            # update generic object data (lat/lon/speed/direction)
            row = update_object_data(det, object_data, internal_id, camera_lat, camera_lon, camera_alt,
                                     camera_facing_deg, camera_roll_deg, camera_pitch_deg, calibration_yaw_deg,
                                     fps, speed_window, frame_idx, class_names)
            framewise_rows.append(row)

            # draw annotations
            draw_annotations(frame, det, internal_id, det['class'], row['speed_kmh'], row['direction_deg'],
                             row['lat'], row['lon'], camera_intrinsics, show_id, show_class, show_direction,
                             show_speed, show_latlon, class_names)

        # write frame and display
        if video_writer is not None and save_video_flag:
            video_writer.write(frame)
        try:
            cv2.imshow('Tracking (Polygon ROI)', cv2.resize(frame, None, fx=0.7, fy=0.7))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        except Exception:
            pass

        frame_idx += 1

    # finalize: summary + save
    df_summary = create_summary_dataframe(object_data)
    safe_save_excel(df_summary, summary_excel)
    if framewise_rows and save_framewise_flag:
        df_fw = pd.DataFrame(framewise_rows)
        safe_save_excel(df_fw, framewise_excel)
    else:
        print("No framewise rows to save.")

    # cleanup
    cap.release()
    if video_writer is not None:
        video_writer.release()
    if device == 'cuda':
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    cv2.destroyAllWindows()
    print("Finished processing.")

if __name__ == "__main__":
    main()