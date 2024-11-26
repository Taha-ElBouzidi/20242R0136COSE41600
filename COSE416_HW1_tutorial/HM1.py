import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import cv2

# Configuration: Enable video generation
create_video_output = False

# Scenario and data path setup
scenario_name = "01_straight_walk"
data_path = f"./data/{scenario_name}/pcd/"
output_frame_dir = f"./output_frames/{scenario_name}/"
output_video_path = f"./output_video/{scenario_name}.mp4"
output_image_path = f"./output_image/{scenario_name}_result.png"

# Ensure directories exist
os.makedirs(output_frame_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

# Load all PCD files
print("Loading PCD files...")
pcd_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".pcd")])
pcd_data_list = [o3d.io.read_point_cloud(file) for file in tqdm(pcd_files)]


def preprocess_point_cloud(point_cloud, voxel_size=0.25):
    """Downsample and clean point cloud data."""
    downsampled = point_cloud.voxel_down_sample(voxel_size)
    _, inliers = downsampled.remove_radius_outlier(nb_points=6, radius=1.2)
    filtered = downsampled.select_by_index(inliers)
    plane_model, inliers = filtered.segment_plane(distance_threshold=0.07, ransac_n=3, num_iterations=1500)
    non_ground_points = filtered.select_by_index(inliers, invert=True)
    return non_ground_points


def detect_motion(previous_pcd, current_pcd, movement_threshold=0.2):
    """Identify moving points between two point clouds."""
    prev_tree = o3d.geometry.KDTreeFlann(previous_pcd)
    moving_points = []

    for point in np.asarray(current_pcd.points):
        _, _, distances = prev_tree.search_knn_vector_3d(point, 1)
        if distances[0] > movement_threshold:
            moving_points.append(point)

    moving_pcd = o3d.geometry.PointCloud()
    moving_pcd.points = o3d.utility.Vector3dVector(np.array(moving_points))
    return moving_pcd


def cluster_and_filter_moving_objects(moving_pcd, clustering_params, bbox_filters):
    """Cluster moving points and filter clusters."""
    labels = np.array(moving_pcd.cluster_dbscan(eps=clustering_params['eps'],
                                                min_points=clustering_params['min_points'],
                                                print_progress=False))
    clusters = []
    for cluster_id in range(labels.max() + 1):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_pcd = moving_pcd.select_by_index(cluster_indices)
        points = np.asarray(cluster_pcd.points)
        x_range = points[:, 0].ptp()
        y_range = points[:, 1].ptp()
        z_range = points[:, 2].ptp()

        if (bbox_filters['min_height'] <= z_range <= bbox_filters['max_height'] and
                bbox_filters['min_width'] <= x_range <= bbox_filters['max_width'] and
                bbox_filters['min_depth'] <= y_range <= bbox_filters['max_depth']):
            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            bbox.color = (1, 0, 0)  # Red for valid bounding boxes
            clusters.append(bbox)
    return clusters


def visualize_point_cloud_with_bboxes(pcd, bboxes, save_path=None, point_size=1.0):
    """Visualize the point cloud with bounding boxes."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Detected Pedestrians")
    vis.add_geometry(pcd)
    for bbox in bboxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    if save_path:
        vis.capture_screen_image(save_path)
        print(f"Visualization saved to {save_path}")
    vis.destroy_window()


# Processing pipeline
previous_pcd = None
merged_point_cloud = o3d.geometry.PointCloud()
all_bounding_boxes = []

for frame_idx, current_pcd in tqdm(enumerate(pcd_data_list), desc="Processing frames"):
    processed_pcd = preprocess_point_cloud(current_pcd)

    if frame_idx % 5 == 0:
        if previous_pcd is not None:
            moving_pcd = detect_motion(previous_pcd, processed_pcd)
            bbox_params = {'eps': 0.5, 'min_points': 8}
            bbox_filters = {
                'min_height': 0.4, 'max_height': 2.5,
                'min_width': 0.3, 'max_width': 1.2,
                'min_depth': 0.3, 'max_depth': 1.2
            }
            bounding_boxes = cluster_and_filter_moving_objects(moving_pcd, bbox_params, bbox_filters)
            all_bounding_boxes.extend(bounding_boxes)

    if frame_idx % 5 == 0:
        previous_pcd = processed_pcd

visualize_point_cloud_with_bboxes(merged_point_cloud, all_bounding_boxes, save_path=output_image_path)