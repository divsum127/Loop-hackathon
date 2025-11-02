from config import *
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
import SimpleITK as sitk
import ipywidgets as widgets
from IPython.display import display, clear_output
import skimage
from skimage import measure, draw
from skimage.draw import polygon2mask
from skimage.util import random_noise
import cv2
from scipy.ndimage import center_of_mass, zoom
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import binary_fill_holes
from scipy.spatial.transform import Rotation as R
from skimage.measure import marching_cubes
from sklearn.cluster import DBSCAN
from ipywidgets import interact, IntSlider
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.signal import savgol_filter
from noise import pnoise3
import copy
import scipy.ndimage as ndi
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
import hydra
import os
import torch
import omegaconf
import matplotlib.pyplot as plt
import os
# from ct_viewer import CTViewer
os.chdir("../../qct_cache_utils/notebooks/")
from ct_viewer import CTViewer
os.chdir("../../src/fake_nodule_3d/")
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go





def vis(array, size = 8, title: str = ""):
        '''
        visualize image from slice to slice using a slider
        array: SimpleITK array
        '''
        num_slices = array.shape[0]

        slider = widgets.IntSlider(value=num_slices//2, min=0, max=num_slices-1, step=1, description='Z Slice')

        def update(slice_index):
            plt.figure(figsize=(size,size))
            plt.imshow(array[slice_index, :, :], cmap='gray')
            plt.title(f"{title} Z Slice {slice_index}")
            #plt.axis('off')
            plt.colorbar(label="HU")
            plt.show()

        out = widgets.interactive_output(update, {'slice_index': slider})
        display(slider, out)


def play_volume(image, interval=0.1, in_=0, out_=200, loop=False):
        """
        Simulate video playback of 3D volume slices.
        
        Parameters:
        - image: SimpleITK array
        - interval: Time in seconds between frames (e.g., 0.1 = 10 fps)
        - loop: Whether to loop the animation indefinitely
        """
        array= image
        num_slices = array.shape[0]

        try:
            while True:
                for z in range(num_slices):
                    if (z>=in_ and z<=out_):
                        clear_output(wait=True)
                        plt.figure(figsize=(16*image.shape[1]/(image.shape[1] + image.shape[2]), 16*image.shape[2]/(image.shape[1] + image.shape[2])))
                        plt.imshow(array[z, :, :], cmap='gray')
                        plt.title(f"Z Slice {z}")
                        plt.axis('off')
                        plt.colorbar(label="HU")
                        plt.show()
                        time.sleep(interval)
                if not loop:
                    break
        except KeyboardInterrupt:
            print("Playback interrupted.")



def visualize_base(points, title=""):

    '''
    a function to visualize 3d shapes interactively
    points: array of boundary point coordinates of the 3d shape. arrays size (K, 3) where K is number of boudary points
    '''

    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='blue',
            opacity=0.3
        )
    )])

    fig.update_layout(
        title=title,
        width= 500,
        height= 500,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()    


def visualize_boundaries_with_slider(boundary_points, volume_shape=(128, 128, 128)):

    '''
    visualize 3d structure in 2d slices using a slider
    boundary_points- array of boundary point coordinates of the 3d shape. arrays size (K, 3) where K is number of boudary points
    '''
    interact(lambda z: show_boundary_slice(z, boundary_points, volume_shape), z=IntSlider(min=0, max=volume_shape[0]-1, step=1, value=volume_shape[0]//2))


def vis_overlay(base, overlay, size=8, title="", alpha=0.5, cmap_overlay='hot', axis=0):
    """
    Visualize 3D base + overlay with an ipywidgets slider in Jupyter.
    GENERALLY USED TO VISUALIZE HOLLOWNESS MAP IN A 3D STRUCTURE 
    BASE- BASE IMAGE ARRAY (SIMPLEITK)
    OVERLAY- HOLLOWNESS MAP ARRAY OF SIZE SAME AS THAT OF BASE
    Parameters:
        base: 3D numpy array (e.g. CT or tissue)
        overlay: 3D numpy array (same shape as base, e.g. hollowness map)
        size: size of figure
        title: plot title prefix
        alpha: transparency of overlay
        cmap_overlay: colormap used for overlay
        axis: 0=z (default), 1=y, 2=x
        
    """
    
    base = np.nan_to_num(base)
    overlay = np.nan_to_num(overlay)

    if axis == 1:
        base = np.transpose(base, (1, 0, 2))
        overlay = np.transpose(overlay, (1, 0, 2))
    elif axis == 2:
        base = np.transpose(base, (2, 0, 1))
        overlay = np.transpose(overlay, (2, 0, 1))

    num_slices = base.shape[0]

    slider = widgets.IntSlider(value=num_slices // 2, min=0, max=num_slices - 1, step=1, description='Slice')

    def update(slice_index):
        plt.figure(figsize=(size, size))
        plt.imshow(base[slice_index], cmap='gray')
        plt.imshow(overlay[slice_index], cmap=cmap_overlay, alpha=alpha)
        plt.title(f"{title} Slice {slice_index}")
        plt.colorbar(label="Overlay Intensity")
        # plt.axis('off')
        plt.show()

    out = widgets.interactive_output(update, {'slice_index': slider})
    display(slider, out)





def generate_sphere(num_pts, image_size= image_size, radius=sphere_r):
        """
        num_pts: number of boundary points we want for the sphere
        image_size: all following masks with be of size {image size}
        radius: radius of sphere
        Generate a set of points on the surface of a 3D sphere.
        """

        if isinstance(image_size, int):
            image_size = (image_size, image_size, image_size)

        cx, cy, cz = np.array(image_size) / 2

        phi = np.arccos(1 - 2 * np.linspace(0, 1, num_pts))
        theta = 2 * np.pi * np.linspace(0, 1, num_pts)

        phi, theta = np.meshgrid(phi, theta)

        x = cx + radius * np.sin(phi) * np.cos(theta)
        y = cy + radius * np.sin(phi) * np.sin(theta)
        z = cz + radius * np.cos(phi)

        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        return np.vstack((x, y, z)).T


def generate_ellipsoid(num_pts, image_size=image_size, rx=40, ry=30, rz=20):
        """
        Generate a set of points on the surface of a 3D ellipsoid.

        Parameters:
        - num_pts: approximate number of surface points
        - image_size: size of the 3D volume (int or tuple)
        - rx, ry, rz: radii along x, y, and z axes

        Returns:
        - N x 3 array of (x, y, z) surface coordinates
        """
        if isinstance(image_size, int):
            image_size = (image_size, image_size, image_size)

        cx, cy, cz = np.array(image_size) / 2

        phi = np.arccos(1 - 2 * np.linspace(0, 1, num_pts))
        theta = 2 * np.pi * np.linspace(0, 1, num_pts)
        phi, theta = np.meshgrid(phi, theta)

        x = rx * np.sin(phi) * np.cos(theta)
        y = ry * np.sin(phi) * np.sin(theta)
        z = rz * np.cos(phi)

        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        rot = R.random().as_matrix()
        rotated_points = points @ rot.T
        rotated_points += np.array([cx, cy, cz])

        # x = x.flatten()
        # y = y.flatten()
        # z = z.flatten()

        return rotated_points # np.vstack((x, y, z)).T        


def generate_superellipsoid(num_pts, image_size=image_size, rx=20, ry=30, rz=40, eps1=1.0, eps2=1.0):
            
        """
    Generate a set of points on the surface of a 3D superellipsoid.

    A superellipsoid is a generalization of an ellipsoid with adjustable shape parameters.
    The surface shape is controlled by the exponents `eps1` and `eps2`, which define how 
    rounded or boxy the shape is along different axes.

    Parameters:
    - num_pts (int): Approximate number of surface points along each angular dimension.
    - image_size (int or tuple): Size of the 3D volume. If int, a cubic volume is assumed.
    - rx (float): Radius along the x-axis.
    - ry (float): Radius along the y-axis.
    - rz (float): Radius along the z-axis.
    - eps1 (float): Shape exponent in the vertical direction (φ). Controls "squareness" of elevation.
                    Values >1 make the shape more box-like; values <1 make it more rounded.
    - eps2 (float): Shape exponent in the horizontal direction (θ). Controls "squareness" of azimuth.

    Returns:
    - points (np.ndarray): Array of shape (N, 3), where N ≈ num_pts^2. Each row is a 3D point (x, y, z)
                           on the surface of the rotated and centered superellipsoid.
    """
    
    
        if isinstance(image_size, int):
            image_size = (image_size, image_size, image_size)
        cz, cy, cx = np.array(image_size) / 2

        n = num_pts
        theta = np.linspace(-np.pi, np.pi, n)
        phi = np.linspace(-np.pi / 2, np.pi / 2, n)
        theta, phi = np.meshgrid(theta, phi)

        def cos_e(angle, e):
            return np.sign(np.cos(angle)) * (np.abs(np.cos(angle)) ** e)

        def sin_e(angle, e):
            return np.sign(np.sin(angle)) * (np.abs(np.sin(angle)) ** e)

        x = rx * cos_e(phi, eps1) * cos_e(theta, eps2)
        y = ry * cos_e(phi, eps1) * sin_e(theta, eps2)
        z = rz * sin_e(phi, eps1)

        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        rot = R.random().as_matrix()
        rotated_points = points @ rot.T
        rotated_points += np.array([cx, cy, cz])

        return rotated_points
        
def generate_radial_noise_shape(num_pts, image_size = image_size):
        """
        Start from a circle and add only low‐frequency radial noise.
        """
        cx, cy, cz = image_size / 2, image_size / 2
        R = random.uniform(25, 35)
        theta = np.linspace(0, 2*np.pi, num_pts, endpoint=False)
        r = np.ones_like(theta) * R
        # Add a few low‐frequency Fourier modes
        for k in (1, 2, 3):
            a_k = np.random.normal(scale=0.08 * R)
            b_k = np.random.normal(scale=0.08 * R)
            r += a_k * np.cos(k * theta) + b_k * np.sin(k * theta)
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        return np.vstack([x, y]).T

def shrink_tissue(vol: np.array, new_shape):

        bg_val = vol.min()
        zoom_factors = np.array(new_shape) / np.array(vol.shape)
        shrunk = zoom(vol, zoom_factors, order=1)  # or 0 if binary
        output = np.full(vol.shape, bg_val, dtype=vol.dtype)
        pad_start = [(p - s) // 2 for p, s in zip(vol.shape, new_shape)]
        pad_end = [start + s for start, s in zip(pad_start, new_shape)]
        output[pad_start[0]:pad_end[0],pad_start[1]:pad_end[1],pad_start[2]:pad_end[2]] = shrunk  

        return output 


def get_radius_for_theta(mask, theta, max_radius= sphere_r):
        '''
        given a binary mask and a  theta values, returns maximum radius in that direction
        mask: 2d binary mask
        theta: angle in radians
        max_radius: sphere radius
        '''

        height, width = mask.shape
        cx, cy = width / 2, height / 2  # image center
        if max_radius is None:
            max_radius = int(np.hypot(cx, cy))
        dx = np.cos(theta)
        dy = np.sin(theta)
        for r in range(max_radius, 0, -1):
            x = int(round(cx + r * dx))
            y = int(round(cy + r * dy))

            if 0 <= x < width and 0 <= y < height:
                if mask[y, x] == 1:
                    return np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        return np.nan  # if no 1 found


def compute_compactness_2d(binary_mask):
        '''
        return compactness value given a 2d binary mask.
        compactness value= 4.pi.(cross sectional area)/(perimeter**2)
        '''
        binary_mask = (binary_mask > 0).astype(np.uint8)
        area = np.sum(binary_mask)
        perimeter = skimage.measure.perimeter_crofton(binary_mask, directions=4)
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        #print(compactness)
        return compactness

def compute_compactness_3d(binary_mask):

        comps = []
        binary_mask = (binary_mask > 0).astype(np.uint8)
        for z in range(binary_mask.shape[0]):
            area = np.sum(binary_mask[z])
            perimeter = skimage.measure.perimeter_crofton(binary_mask[z], directions=4)
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            comps.append(compactness)
        comps = np.array(comps)
        comps = comps[~np.isnan(comps)]    
        return comps.mean()



def find_flow_params(target_compactness, tissue_type, nodule_type, tolerance=compactness_mean_tolerance):

        df = pd.read_csv(flow_data_csv)
        std_dev = PARAMS[tissue_type]["compactness_std_dev"]
        mask = df['mean_compactness'].between(target_compactness - tolerance, target_compactness + tolerance)
        filtered_df2 = df[mask]
        filtered_df = filtered_df2.copy()
        filtered_df['sigma_diff'] = filtered_df['sigma_weight'] - filtered_df['sigma_ctrl']
        max_diff_row = filtered_df.loc[filtered_df['sigma_diff'].idxmax()]

        if nodule_type == "spiculated":
            return {
                'sigma_ctrl': max_diff_row['sigma_ctrl'],
                'sigma_weight': max_diff_row['sigma_weight'],
                'num_steps': int(max_diff_row['num_steps'])
            }
            

        strict_df = filtered_df[filtered_df['std_compactness'] < std_dev]
        if not strict_df.empty:
            candidates = strict_df
        else:
            min_std_row = filtered_df.loc[filtered_df['std_compactness'].idxmin()]
            candidates = filtered_df[filtered_df['std_compactness'] == min_std_row['std_compactness']]
        
        best_row = candidates.loc[candidates['num_steps'].idxmin()]

        flow = {
            'sigma_ctrl': best_row['sigma_ctrl'],
            'sigma_weight': best_row['sigma_weight'],
            'num_steps': int(best_row['num_steps'])
        }

        return flow                


def boundary_to_mask(boundary_points, volume_shape= grid_size):

            '''
            boundary_points: list of cartesian coordinates for all of boundary points (3d)
            returns the corrosponding 3d binary mask 
            '''
            mask_3d = np.zeros(volume_shape, dtype=np.uint8)

            points = np.round(boundary_points).astype(int)
            points = np.clip(points, 0, np.array(volume_shape[::-1]) - 1)

            for z in np.unique(points[:, 2]):
                z = int(z)
                slice_points = points[points[:, 2] == z][:, :2]

                if len(slice_points) < 3:
                    continue

                img = np.zeros((volume_shape[1], volume_shape[2]), dtype=np.uint8)

                contours = [slice_points.reshape(-1, 1, 2).astype(np.int32)]
                cv2.fillPoly(img, pts=contours, color=1)
                mask_3d[z] = img

            return mask_3d


def insert_holes_random(volume, euler_num, sphere_r= sphere_r, hole_r= hole_r):

        '''
        iserts holes at random positions with centre within 0.75 times sphere_r. uses perlin to add noise
        volume: volume array
        euler_num: euler characteristic defined as 2-2h, where h= number of holes in a structure
        hole_r: hole radius
        '''    
        num_holes = (2 - euler_num) // 2
        volume1 = np.copy(volume)
        sphere_center = np.array(volume.shape) // 2
        
        # Use a truly random seed from system entropy
        rng = np.random.default_rng()  # No seed specified → non-deterministic

        z, y, x = np.indices(volume.shape)
        
        for _ in range(num_holes):
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(0, np.pi)
            inner_r = 0.75 * sphere_r
            r= random.uniform(0, inner_r)
            offset = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ]).astype(int)

            hole_center = sphere_center + offset
            hole_dist = np.sqrt((x - hole_center[2])**2 +
                                (y - hole_center[1])**2 +
                                (z - hole_center[0])**2)
            hole_mask = hole_dist <= hole_r
            volume1[hole_mask] = 0
        
        volume_sitk = sitk.GetImageFromArray(volume1)
        volume_sitk.SetSpacing((1.0, 1.0, 1.0))
        return volume1, volume_sitk



def insert_holes_random_distant(volume, euler_num, sphere_r= sphere_r, hole_r= hole_r, max_attempts=100):
        '''
        Inserts non-overlapping holes at random positions inside a sphere.
        
        Parameters:
        - volume: 3D binary NumPy array (1 inside shape, 0 outside)
        - sphere_r: radius of the original sphere
        - euler_num: desired Euler characteristic (2 - 2h ⇒ h = number of holes)
        - hole_r: radius of each hole
        - max_attempts: max retries per hole to avoid infinite loop

        Returns:
        - Modified volume and its SimpleITK image
        '''
        num_holes = int((2 - euler_num) // 2)
        volume1 = np.copy(volume)
        sphere_center = np.array(volume.shape) // 2
        rng = np.random.default_rng()
        z, y, x = np.indices(volume.shape)
        placed_centers = []

        if num_holes <=0:
                volume_sitk = sitk.GetImageFromArray(volume1)
                volume_sitk.SetSpacing((1.0, 1.0, 1.0))
                return volume1, volume_sitk
        
        for _ in range(num_holes):
            for attempt in range(max_attempts):
                theta = rng.uniform(0, 2 * np.pi)
                phi = rng.uniform(0, np.pi)
                inner_r = 0.75 * sphere_r
                r = rng.uniform(0, inner_r)

                offset = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ])
                hole_center = sphere_center + offset
                hole_center = hole_center.astype(int)

                # Ensure hole_center is within bounds
                if np.any(hole_center - hole_r < 0) or np.any(hole_center + hole_r >= np.array(volume.shape)):
                    continue

                # Check against previous holes
                too_close = False
                for c in placed_centers:
                    if np.linalg.norm(hole_center - c) < 2 * hole_r:
                        too_close = True
                        break
                if too_close:
                    continue

                # No conflict, place the hole
                placed_centers.append(hole_center)
                hole_dist = np.sqrt((x - hole_center[2])**2 +
                                    (y - hole_center[1])**2 +
                                    (z - hole_center[0])**2)
                hole_mask = hole_dist <= hole_r
                volume1[hole_mask] = 0
                break
         

        volume_sitk = sitk.GetImageFromArray(volume1)
        volume_sitk.SetSpacing((1.0, 1.0, 1.0))
        return volume1, volume_sitk


def compute_hollowness(volume, sigma=5):

        mat_mask = (volume > volume.min()*0.95) .astype(np.uint8)
        filled_mask = binary_fill_holes(mat_mask).astype(np.uint8)
        
        volume = mat_mask.astype(np.float32)
        mask = filled_mask.astype(np.float32)

        weighted_sum = gaussian_filter(volume * mask, sigma=sigma, mode='constant', cval=0.0)

        weight_sum = gaussian_filter(mask, sigma=sigma, mode='constant', cval=0.0)

        with np.errstate(divide='ignore', invalid='ignore'):
            hollowness_map = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)

        masked_map = np.where(filled_mask, hollowness_map, np.inf)
        target_center = np.unravel_index(np.argmin(masked_map), masked_map.shape)    
        return hollowness_map, target_center

def find_k_separated_minima(hollow_map, k= number_of_hollow_points_to_find, min_distance= min_distance_between_target_points, smooth_sigma=2):

        nan_mask = np.isnan(hollow_map)
        map_safe = hollow_map.copy()
        max_val = np.nanmax(hollow_map)
        map_safe[nan_mask] = max_val  # temp replace

        smoothed = gaussian_filter(map_safe, sigma=smooth_sigma)

        # Step 3: Reinstate NaNs
        smoothed[nan_mask] = np.nan

        selected = []
        mask = ~nan_mask  # Only valid (non-NaN) locations are True

        for _ in range(k):
            masked_map = np.where(mask, smoothed, np.nanmax(smoothed) + 1)
            min_idx = np.unravel_index(np.nanargmin(masked_map), masked_map.shape)
            min_val = masked_map[min_idx]

            if np.isnan(min_val):
                raise RuntimeError(f"Only found {len(selected)} valid points. Try reducing min_distance.")

            selected.append(min_idx)

            # Mask out spherical region around selected point
            zz, yy, xx = np.indices(smoothed.shape)
            dist = np.sqrt((zz - min_idx[0])**2 + (yy - min_idx[1])**2 + (xx - min_idx[2])**2)
            mask[dist < min_distance] = False

        return selected    


def extract_all_surfaces(volume):
        all_boundaries = []

        # Step 1: Fill holes to extract only outer surface
        verts, _, _, _ = marching_cubes(volume, level=0.5)
        outer_boundary = [tuple(v) for v in verts]
        all_boundaries.append(np.array(outer_boundary))

        # Step 2: Find internal holes
        inverted = ~volume.astype(bool)
        labeled = label(inverted)
        background_label = labeled[0,0,0]
        internal_air = (labeled != background_label) & (labeled > 0)

        # Step 3: Marching cubes on internal cavities
        if np.any(internal_air):
            verts_in, _, _, _ = marching_cubes(internal_air.astype(np.uint8), level=0.5)
            all_boundaries.append(np.array(verts_in))

        return all_boundaries



def show_boundary_slice(z, boundary_points, volume_shape= grid_size):
        points = np.round(boundary_points).astype(int)
        points = np.clip(points, 0, np.array(volume_shape[::-1]) - 1)
        
        slice_points = points[points[:, 2] == z]
        
        plt.figure(figsize=(5, 5))
        if len(slice_points) > 0:
            plt.scatter(slice_points[:, 0], slice_points[:, 1], s=1, c='red')
        plt.title(f'Z Slice {z}')
        plt.xlim(0, volume_shape[2])
        plt.ylim(0, volume_shape[1])
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.grid(True)
        plt.show()



def split_boundary_into_holes(boundary_points, eps=1, min_samples=5):
        """
        Splits a set of 3D boundary points into separate groups (holes) using DBSCAN.
        
        Parameters:
            boundary_points: (N, 3) array of x, y, z points.
            eps: max distance between points in same cluster.
            min_samples: minimum number of points to form a cluster.

        Returns:
            List of (M_i, 3) arrays, one per detected hole.
        """
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(boundary_points)
        labels = clustering.labels_

        hole_boundaries = []
        for label in np.unique(labels):
            if label == -1:
                continue  # skip noise
            mask = labels == label
            hole_boundaries.append(boundary_points[mask])

        return hole_boundaries        



