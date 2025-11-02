from nodule import *
from config import *
from helper_functions import *
import config
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
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go





class Nodule():

    def __init__(self):

        self.tissue_mat = {
            "Active_Fibrosis": None,
            "Cancer_BAC": None,
            "Cancer_solid": None,
            "Inactive_Fibrosis": None,
            "Necrosis": None,
            "RBC": None
        } 

        self.type= None
        self.base = "spherical"
        self.nodule_mat = None
        
        if self.type == ("polygonal" or "tentacular"):
            self.base = "superelipsoidal"

        self.data_csv = flow_data_csv


    def modify_params(self, tissue_type):
        '''
        Tissue: Tissue for which we need new params. Will be used while merging all tissues
        tissue_type (str): "Cancer_solid" / "Cancer_BAC" / "Inactive_Fibrosis" / "Active_Fibrosis" / "Necrosis" / "RBC" supported
        '''
        mean_HU = PARAMS[tissue_type]["mean_HU"]
        std_HU = PARAMS[tissue_type]["std_HU"]
        sphere_r = PARAMS[tissue_type]["sphere_r"]
        euler_num = PARAMS[tissue_type]["euler_num"]
        hole_r = PARAMS[tissue_type]["hole_r"]
        volume_shape = PARAMS[tissue_type]["volume_shape"]
        target_compactness = PARAMS[tissue_type]["target_compactness"]
        compactness_std_dev = PARAMS[tissue_type]["compactness_std_dev"]
        hu_thresh = PARAMS[tissue_type]["hu_thresh"]
        
    

    def apply_flow_3d(self, boundary, flow, weights= None):
            '''
            applies 3d flow for a 3d boundary
            Apply a single ODE‐based deformation with Gaussian‐smoothed random velocities.
            Recomputes the Gaussian kernel at each time‐step so deformation remains visible.
            Returns a list of boundary arrays at each time‐step (including initial).
            boundary: list of cartesian coordinates of all boundary points
            flow: {
                sigma_ctrl: range of deformation for each boundary point
                sigma_weight: more the sigma_weight, more is the deformation for each boudnary point
                num =_steps: total 
            }
            weights: weights for boundary deformation
            '''

            ctrl_grid= (num_ctrl_x, num_ctrl_y, num_ctrl_z)
            sigma_ctrl   = flow['sigma_ctrl']
            sigma_weight = flow['sigma_weight']
            n_steps      = int(flow['num_steps'])
    
            dt = 1.0 / n_steps
            
            min_xyz = boundary.min(axis=0) - 5
            max_xyz = boundary.max(axis=0) + 5

            ctrl_x = np.linspace(min_xyz[0], max_xyz[0], ctrl_grid[0])
            ctrl_y = np.linspace(min_xyz[1], max_xyz[1], ctrl_grid[1])
            ctrl_z = np.linspace(min_xyz[2], max_xyz[2], ctrl_grid[2])

            ctrl_centers = np.array([(x, y, z) for z in ctrl_z for y in ctrl_y for x in ctrl_x])

            K= len(ctrl_centers)
            if weights is not None:
                pass
            else:
                weights = np.random.normal(scale=sigma_weight, size=(int(n_steps), K, 3))

            all_boundaries = [boundary.copy()]
            current = boundary.copy()
            
            for t in range(n_steps):
                diff = current[:, np.newaxis, :] - ctrl_centers[np.newaxis, :, :]  # (num_pts, K, 2)
                dist_sq = np.sum(diff**2, axis=2)                                    # (num_pts, K)
                gauss_kernel = np.exp(-dist_sq / (2 * sigma_ctrl**2))               # (num_pts, K)
                
                v = gauss_kernel.dot(weights[t])   # (num_pts, 2)
                
                current = current + dt * v
                all_boundaries.append(current.copy())
                # print(f"step {t} done") 
                
            return all_boundaries[-1], weights    
                
            

    def smooth_boundary(self, boundary, window=51, polyorder=3):
        """
        Smooth 3D boundary points using Savitzky–Golay filtering.

        Parameters:
        - boundary (np.ndarray): (N, 3) array of boundary coordinates.
        - window (int): Window size for smoothing (must be odd and <= N).
        - polyorder (int): Polynomial order used in filtering.

        Returns:
        - (N, 3) np.ndarray of smoothed boundary coordinates.
        """
        N = boundary.shape[0]
        if N < window:
            return boundary
        x = boundary[:,0]
        y = boundary[:,1]
        z = boundary[:, 2]
        x_s = savgol_filter(x, window_length=window, polyorder=polyorder, mode='wrap')
        y_s = savgol_filter(y, window_length=window, polyorder=polyorder, mode='wrap')
        z_s = savgol_filter(z, window_length=window, polyorder=polyorder, mode='wrap')

        return np.vstack([x_s, y_s, z_s]).T



    def merge_tissues_at_hollow(self, tissue1: np.array, tissue2: np.array, insert_center= (64, 64, 64)):
            """
            Merge two 3D tissue volumes by inserting `tissue2` into a low-density 
            region of `tissue1` at a specified location, and blending the overlap.

            Parameters:
            - tissue1 (np.ndarray): Base 3D tissue volume (Cancer_solid used here since it has the highest volumetric composition).
            - tissue2 (np.ndarray): 3D tissue volume of the one to be inserted.
            - insert_center (tuple): Coordinate (z, y, x) where `tissue2` will be merged into `tissue1`.

            Returns:
            - np.ndarray: Cropped and blended 3D tissue volume of same shape as `tissue1`.
            """
            base_shape = tissue1.shape
            base_center = base_shape[0] // 2, base_shape[1] // 2, base_shape[2] // 2
            buffer = base_center[0]
            canvas_shape = 2*(tissue1.shape[0]), 2*(tissue1.shape[1]), 2*(tissue1.shape[2])
            canvas_center = canvas_shape[0] // 2, canvas_shape[1] // 2, canvas_shape[2] // 2
            canvas = np.full(canvas_shape, fill_value=-55, dtype=np.float32)

            canvas[canvas_center[0]- buffer : canvas_center[0]+ buffer, canvas_center[1]- buffer : canvas_center[1]+ buffer, canvas_center[2]- buffer : canvas_center[2]+ buffer] = tissue1
            insert_canvas_center = (canvas_center[0] + insert_center[0]- base_center[0], canvas_center[1] + insert_center[1]- base_center[1], canvas_center[2] + insert_center[2]- base_center[2])
            region = canvas[insert_canvas_center[0]- buffer : insert_canvas_center[0]+ buffer, insert_canvas_center[1]- buffer : insert_canvas_center[1]+ buffer, insert_canvas_center[2]- buffer : insert_canvas_center[2]+ buffer]
            raw_mask = tissue2 > -54
            filled_mask = binary_fill_holes(raw_mask)
            blended = region.copy()
            blended[filled_mask] = 0.5 * region[filled_mask] + 0.5 * tissue2[filled_mask]
            canvas[insert_canvas_center[0]- buffer : insert_canvas_center[0]+ buffer, insert_canvas_center[1]- buffer : insert_canvas_center[1]+ buffer, insert_canvas_center[2]- buffer : insert_canvas_center[2]+ buffer] = blended

            binary_mask = canvas > -55
            com = center_of_mass(binary_mask)
            cz, cy, cx = map(int, com)

            crop_size = base_shape  # target size (128,128,128)
            z_start = max(0, cz - crop_size[0] // 2)
            y_start = max(0, cy - crop_size[1] // 2)
            x_start = max(0, cx - crop_size[2] // 2)
            z_end = z_start + crop_size[0]
            y_end = y_start + crop_size[1]
            x_end = x_start + crop_size[2]

            cropped = canvas[z_start:z_end, y_start:y_end, x_start:x_end]

            # 5. If cropped size ≠ final_shape (edge case), resample using zoom
            if cropped.shape != base_shape:
                zoom_factors = [f / c for f, c in zip(base_shape, cropped.shape)]
                cropped = zoom(cropped, zoom=zoom_factors, order=1)  # linear interpolation

            return cropped    
    



    def contrast_profile(self, vol_data, C=mean_HU + std_HU, B=hu_thresh+7, n1=2.0, n2=2.4, sphere_r=sphere_r):
        """
        Apply a spatial intensity profile across a nodule volume based on distance from center.

        Parameters:
        - vol_data (np.ndarray): Binary 3D volume (shape: 128×128×128) representing the nodule.
        - C (float): Peak intensity at the nodule center.
        - B (float): Background base intensity value.
        - n1 (float): Power exponent at the central axial slice (defines peak falloff).
        - n2 (float): Power exponent at slices farther from the center (steeper falloff).
        - sphere_r (float): Reference radius of the sphere for profile shaping.

        Returns:
        - np.ndarray: 3D volume with intensity values modulated to simulate internal contrast variation.
        
        Notes:
        - For each voxel marked as nodule (value == 1), the local radial distance is used
        to compute intensity using: `C * ((1 - (r / R(θ))²)ⁿ) + B`.
        - Exponent `n` varies slice-wise between `n1` and `n2` to mimic axial attenuation.
        - Adds random and Perlin noise at the end to improve realism.
        """

        vol_data = np.asarray(vol_data, dtype=np.float32)
        
        final_mask = np.zeros_like(vol_data, dtype=np.float32)

        depth, height, width = vol_data.shape
        cx, cy = width / 2, height / 2 

        for z in range(depth):

            slice_ = vol_data[z]
            mask = np.full_like(slice_, fill_value=hu_thresh-5, dtype=np.float32)
            ys, xs = np.where(slice_ == 1.0)  

            if z>=29 and z<=98:
                for y, x in zip(ys, xs):
                    dx = x - cx
                    dy = y - cy
                    theta = np.arctan2(dy, dx)
                    theta = (theta + 2 * np.pi) % (2 * np.pi)

                    radius = get_radius_for_theta(slice_, theta)
                    if np.isnan(radius) or radius <= 0:
                        continue
                    f= np.sqrt(sphere_r**2 - np.abs(z- cx)**2)/sphere_r
                    n = ((n2 - n1) * 6 / 5) * (1 - f) + n1
                    r = np.sqrt(dx**2 + dy**2)

                    base = np.clip(1 - (r / radius)**2, 0, None)
                    mask[int(y), int(x)] = C * (base**n) + B


            final_mask[z] = mask

            #print(f"Slice {z} done")
        contrast_vol = self.add_random_noise(final_mask)
        contrast_vol = self.add_perlin_noise_3d(sitk.GetArrayFromImage(contrast_vol))
        return contrast_vol



    def add_random_noise(self, vol_data_arr, var= 0.01):
        '''
        adds random noise after contrast distribution
        vol_data_arr: volume array
        var: variance for the random noise
        '''

        final_vol= np.copy(vol_data_arr)

        for z in range(len(vol_data_arr[0])):

            image = vol_data_arr[z].astype(np.float32)
            image = np.nan_to_num(image, nan=-55) 
            p1, p99 = np.percentile(image, 1), np.percentile(image, 99)
            image_clipped = np.clip(image, p1, p99)
            image_min, image_max = image_clipped.min(), image_clipped.max()
            denominator = image_max - image_min
            if denominator == 0:
                image_norm = np.zeros_like(image_clipped)
            else:
                image_norm = (image_clipped - image_min) / denominator
            noisy_norm = random_noise(image_norm, mode="gaussian", var=0.002)
            noisy = noisy_norm * (image_max - image_min) + image_min
            mask = image > -54
            result = np.copy(image)
            result[mask] = noisy[mask]
            final_vol[z] = result
            final_vol_img = sitk.GetImageFromArray(final_vol)
            
        return final_vol_img  


    def add_perlin_noise_3d(self, volume, scale=0.1, amplitude=30, seed= np.random.randint(0, 10)
    ):
        '''
        Uses pnoise3d to add perlin noise after random noise to the contrast profile
        volume: volume array
        scale: scale
        amplitude: the weight given to perlin noise after contrast profile applied. 
        more the amplitude, more the intensity distribution goes towards perlin distribution rather than original contrast distribution
        '''
    
        shape = volume.shape
        noisy_volume = np.full_like(volume, -55, dtype=np.float32)

        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    noise_val = pnoise3(
                        x * scale, y * scale, z * scale,
                        octaves=8,
                        persistence=0.5,
                        lacunarity=2.0,
                        repeatx=shape[2],
                        repeaty=shape[1],
                        repeatz=shape[0],
                        base=seed
                    )
                    if volume[z,y,x] >-52:
                        noisy_volume[z, y, x] = volume[z, y, x] + amplitude * noise_val
                        noisy_volume[z,y,x] = np.clip(noisy_volume[z,y,x], hu_thresh, mean_HU+std_HU)


        return noisy_volume




    def merge_lobulated_tissues(self, tissue_mat):
        """
        Merge multiple tissue types into a single 3D volume in a spatially distributed, hollow-guided manner.

        Parameters:
        - tissue_mat (dict): Dictionary mapping tissue names to their corresponding 3D volume arrays. 
        Must include keys: "Cancer_solid", "Cancer_BAC", "Inactive_Fibrosis", 
        "Active_Fibrosis", "Necrosis", "RBC".

        Returns:
        - np.ndarray: Final 3D volume (shape: 128×128×128) with all tissues merged around 
        hollowness-guided target centers.

        Method:
        1. Select "Cancer_solid" as the base tissue.
        2. Compute a hollowness map of the base to find 5 spatially separated low-density regions.
        3. Shrink all tissue types proportionally based on their intensity alpha relative to "Cancer_solid".
        4. Iteratively merge each shrunk tissue into the base at the selected hollow targets (allot more hollow spots to the tissues which are supposed to have greater composition).
        5. Final merge ensures spatial separation and compact integration of all tissue types.
        """

        base_tissue = tissue_mat["Cancer_solid"]
        merged_mat= base_tissue

        hollowness_map, __= compute_hollowness(base_tissue)
        target_centers = find_hollow_target_centers(hollow_map= hollowness_map, k=5, min_distance= 20)

        relative_shapes = {k: int((v / alphas["Cancer_solid"])*128) for k, v in alphas.items()}
        shrinked_tissues = tissue_mat.copy()
        shrinked_tissues = {k: shrink_tissue(vol= v, new_shape= (relative_shapes[k], relative_shapes[k], relative_shapes[k])) for k, v in shrinked_tissues.items()}

        merged1 = self.merge_tissues_at_hollow(base_tissue, shrinked_tissues["Cancer_BAC"], target_centers[0])
        merged2 = self.merge_tissues_at_hollow(merged1, shrinked_tissues["Inactive_Fibrosis"], target_centers[1])
        merged3 = self.merge_tissues_at_hollow(merged2, shrinked_tissues["Active_Fibrosis"], target_centers[2])
        merged4 = self.merge_tissues_at_hollow(merged3, shrinked_tissues["Necrosis"], target_centers[3])
        merged_final = self.merge_tissues_at_hollow(merged4, shrinked_tissues["RBC"], target_centers[4])

        return merged_final
            

    def tissue_pipeline(self, tissue_type):
        """
        Master pipeline to synthesize a 3D tissue volume of a specific type by orchestrating
        all major submodules: shape generation, deformation, hole insertion, contrast modeling,
        and final volume construction.

        Parameters:
        - tissue_type (str): The category of tissue to generate (e.g., "Cancer_solid", "Inactive_Fibrosis").
        Returns:
        - np.ndarray: A synthesized 3D volume of shape (128, 128, 128) representing the target tissue.

        Workflow Overview:
        1. **Base Shape Initialization**:
        Depending on `self.type`, generate a base geometry:
        - 'round', 'spiculated', 'lobulated' → spherical
        - 'tentacular' → superellipsoid
        - 'ovular' → ellipsoid

        2. Initial Surface Extraction:
        Convert base boundary points into a 3D binary mask, extract its surface.

        3. Topological Modification:
        For spiculated and lobulated types, inject distant topological holes 
        (using `insert_holes_random_distant`) and re-extract surfaces.

        4. Surface Decomposition:
        Split the object surface into individual connected components 
        (`split_boundary_into_holes`), treating them as distinct boundaries.

        5. Deformation with Flow:
        For each surface component:
        - Fetch optimal deformation parameters via `find_flow_params`.
        - Apply 3D flow deformation using `apply_flow_3d`.
        - For all but the first, scale flow and subtract resulting mask.

        6. Contrast Modeling:
        Apply the intensity profile (`contrast_profile`) to convert binary mask 
        into a realistic HU-distributed tissue structure.

        7. Storage and Return:
        Store final volume in `self.tissue_mat[tissue_type]` and return it.
        """

        sphere= None
     
        if self.type == "round":
            sphere = generate_sphere(300, 128)

        if self.type == "spiculated":
            sphere = generate_sphere(300, 128)

        if self.type == "lobulated":
            sphere = generate_sphere(300, 128)        
        
        if self.type == "tentacular":
            sphere = generate_superellipsoid(num_pts= 300, eps1 = 1.5, eps2= 1.5)

        if self.type == "ovular":
            sphere = generate_ellipsoid(num_pts = 300)    
         

 
            
        sphere_bound= boundary_to_mask(sphere)
        base_b = extract_all_surfaces(sphere_bound)

        if self.type == "lobulated" or "spiculated":
            sphere_after_euler, sphere_after_euler_img=  insert_holes_random_distant(sphere_bound, euler_num= euler_num)
            base_b= extract_all_surfaces(sphere_after_euler)
            
        all_boundaries = split_boundary_into_holes(base_b[0])
        all_boundaries[0] = sphere
        flow3d = find_flow_params(target_compactness, tissue_type, self.type)
        final_mask = None
        weights = None


        for boundary in all_boundaries:

            if final_mask is None:
                boundaries, weights = self.apply_flow_3d(boundary, flow3d)
                mask = boundary_to_mask(boundaries)
                mask = binary_fill_holes(mask)
                mask = mask.astype(np.uint8)
                final_mask = mask
                
                #print("1 BOUNDARY DONE")
            else:
                factor = len(boundary)/ len(all_boundaries[0])
                flow_sub= {
                        'sigma_ctrl': flow3d['sigma_ctrl']*factor,
                        'sigma_weight': flow3d['sigma_weight']*factor,
                        'num_steps': flow3d['num_steps']
                        }
                boundaries, _ = self.apply_flow_3d(boundary, flow_sub, weights*factor)
                boundaries = self.smooth_boundary(boundaries)
                boundaries = self.smooth_boundary(boundaries)
                mask= boundary_to_mask(boundaries)
                mask = binary_fill_holes(mask)
                mask = mask.astype(np.uint8)
                final_mask-= mask    
                #print("1 BOUNDARY DONE")
                pass


        contrast_vol = self.contrast_profile(final_mask)
        self.tissue_mat[tissue_type] = contrast_vol
        return contrast_vol


    '''
    blending all tissues (right now without holes)
    '''

    def nodule_pipeline(self, nodule_type):
        '''
        Master function for nodule generation.
        - nodule_type (str): "Round" / "Ovular" / "Lobulated" / "Spiculated" / "Tentacular" supported as of now
        '''

        self.type = nodule_type

        final_tissue_arrays = {
            "Cancer_solid": None,
            "Cancer_BAC": None,
            "Active_Fibrosis": None,
            "Inactive_Fibrosis": None,
            "RBC": None,
            "Necrosis": None
        }
    
        for tissue_type in PARAMS:

            self.modify_params(tissue_type)
            contrast_vol = self.tissue_pipeline(tissue_type)
            final_tissue_arrays[tissue_type] = contrast_vol
            
            
        alphas = {
            "Active_Fibrosis": 0.13,
            "Cancer_BAC": 0.22,
            "Cancer_solid": 0.46,
            "Inactive_Fibrosis": 0.13,
            "Necrosis": 0.03,
            "RBC": 0.03
        }

        if self.type == "lobulated" or "spiculated":
            blend_lobulated = self.merge_lobulated_tissues(self.tissue_mat)
            self.nodule_mat = blend_lobulated
            return blend_lobulated

        blended = sum(alphas[name] * final_tissue_arrays[name]/6 for name in list(final_tissue_arrays.keys()))   
        self.nodule_mat = blended 
        return blended



    def visualize_specific_tissue_contribution(self, tissue: str):
        '''
        In case we want visualization for individual tissue contribution. (Generates nodule by its own)
        (has some bugs- need to fix this. right now its following simple blending strategy by blending all tissues at a common center)
        '''
        
        final_tissue_arrays= self.tissue_mat

        images = self.tissue_mat

        colors = {
            "Active_Fibrosis":    [1.0, 0.6, 0.6],  
            "Cancer_BAC":         [0.6, 1.0, 0.6], 
            "Cancer_solid":       [0.6, 0.75, 1.0], 
            "Inactive_Fibrosis":  [1.0, 0.9, 0.6],  
            "Necrosis":           [0.85, 0.6, 1.0],
            "RBC":                [0.6, 1.0, 1.0],  
        }

        alphas = {
            "Active_Fibrosis": 0.5,
            "Cancer_BAC": 0.5,
            "Cancer_solid": 0.5,
            "Inactive_Fibrosis": 0.5,
            "Necrosis": 0.5,
            "RBC": 0.5,
        }

        highlight_tissue = tissue

        image_names = list(images.keys())
        image_array = np.stack([images[name] for name in image_names])  # (T, Z, H, W)

        image_array = (image_array - image_array.min(axis=(1, 2, 3), keepdims=True)) / \
                    (image_array.max(axis=(1, 2, 3), keepdims=True) + 1e-8)

        T, Z, H, W = image_array.shape
        composite_volume = np.zeros((Z, H, W, 3), dtype=np.float32)

        highlight_idx = image_names.index(highlight_tissue)
        highlight_mask = image_array[highlight_idx] > 0  # shape: (Z, H, W)

        for z in range(Z):
            composite = np.zeros((H, W, 3), dtype=np.float32)
            mask = highlight_mask[z]

            for i, name in enumerate(image_names):
                grayscale = image_array[i, z, :, :]  # (H, W)
                color = np.array(colors[name])
                alpha = alphas[name]
                rgb = grayscale[..., None] * color  # (H, W, 3)

                if name == highlight_tissue:
                    composite[mask] = rgb[mask]
                else:
                    composite[~mask] = composite[~mask] * (1 - alpha) + rgb[~mask] * alpha

            composite_volume[z] = composite
        return composite_volume    
        '''
        composite volume is nodule but with selected tissue highlighted. (can be visualized using vis() or play_volume()...)
        '''



class CTScan():

    def __init__(self):

        self.master_df = None
        self.df = None
        self.aids= None
        self.sids = None
        self.region = 'stats'
        self.nodule = None
        self.sid= None
        self.path = "train_1_a_1.nii"
        self.scans = None
        self.masks= None
        self.bboxes_masks = None
        self.metadata= None
        self.merge_centers= None
        self.nodule_size = None
        self.blend_factor = None
        self.merged_ct = None



    def load_ct(self, path):
      """
      Returns:
        arr: (Z, Y, X) float32 NumPy array
        spacing: (dz, dy, dx) voxel size in mm
        direction: 3x3 orientation cosines
        origin: world-space origin (x,y,z)
      """
      img = sitk.ReadImage(path)                 # .nii or .nii.gz
      arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
      spacing = img.GetSpacing()[::-1]           # SITK gives (dx,dy,dz); reverse to (dz,dy,dx)
      direction = np.array(img.GetDirection()).reshape(3, 3)
      origin = img.GetOrigin()
      self.scans = arr

      if self.masks is not None:
        lung_mask = self.masks
        return arr, lung_mask, spacing, direction, origin
      img = sitk.GetImageFromArray(arr.astype(np.float32))  # SITK expects (Z,Y,X)
    # SITK uses (dx,dy,dz). We were given (dz,dy,dx), so reverse
      img.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))
      img.SetOrigin(origin)
      if isinstance(direction, np.ndarray):
          direction = tuple(direction.reshape(-1).tolist())
      img.SetDirection(direction)

      # Run model (labels: 0 background, 1/2 lungs; sometimes 3 for trachea depending on model)
      seg = lungmask_api.apply(img)   # returns SITK image or numpy? -> numpy array (Z,Y,X)
      # Binarize: lungs are > 0
      lung_mask = (seg > 0).astype(np.uint8)
      self.masks = lung_mask
      return arr, lung_mask, spacing, direction, origin




    def merge_nodule_with_ct(self, ct_mat: np.array, nodule_mat: np.array, location: tuple, alpha: float = 0.5, size = 5):
        """
        Blend a synthetic nodule into a CT scan at the specified 3D location.
        WE ARE NOT APPLYING GAUSSIAN SMOOTHENING FOR SPICULATED NODULES TO PRESERVE THE SPIKES AORUND THE BOUNDARY
        Parameters:
        - ct_mat (np.ndarray): Original CT volume.
        - nodule_mat (np.ndarray): Nodule volume to insert.
        - location (tuple): (z, y, x) center for insertion.
        - alpha (float): Blending weight for smooth merge.
        - size (int): Target patch size for insertion.

        Returns:
        - Updated CT volume with nodule.
        - Intensity values of inserted voxels.
        """
        final_min = -1000
        final_max = 500

        nodule_mat = ((nodule_mat-nodule_mat.min())/(nodule_mat.max()- nodule_mat.min()))*(final_max- final_min) + final_min
        zc, yc, xc = location
        half = size // 2
        nodule_mat = nodule_mat[25:100, :, :]
        nodule_mask = (nodule_mat > -500)

        zoom_factors = np.array([size / s for s in nodule_mat.shape])
        nodule_rescaled = zoom(nodule_mat, zoom=zoom_factors, order=1)
        mask_rescaled = zoom(nodule_mask.astype(float), zoom=zoom_factors, order=0) > 0.5

        z1, z2 = zc - half, zc + half + 1
        y1, y2 = yc - half, yc + half + 1
        x1, x2 = xc - half, xc + half + 1

        ct_mat_cp = ct_mat.copy()

        if self.nodule.type == "spiculated":
            ct_mat_cp[z1:z2, y1:y2, x1:x2][mask_rescaled] =  nodule_rescaled[mask_rescaled]*alpha + (1-alpha)*ct_mat_cp[z1:z2, y1:y2, x1:x2][mask_rescaled]
            return ct_mat_cp, nodule_rescaled[mask_rescaled]

        ct_mat_cp[z1:z2, y1:y2, x1:x2] = self.smooth_blend(ct_patch=ct_mat_cp[z1:z2, y1:y2, x1:x2], nodule_patch=nodule_rescaled, nodule_mask=mask_rescaled, alpha = alpha)
        # ct_mat_cp[z1:z2, y1:y2, x1:x2][mask_rescaled] =  nodule_rescaled[mask_rescaled]*alpha + (1-alpha)*ct_mat_cp[z1:z2, y1:y2, x1:x2][mask_rescaled]
        return ct_mat_cp, nodule_rescaled[mask_rescaled]


    def smooth_blend(self, ct_patch, nodule_patch, nodule_mask, alpha, sigma= 1.5):
        """
        Blend a nodule patch into a CT patch using a Gaussian-smoothed mask to match with the CT envirnment.
        NOT USED FOR SPICULATED NODULE
        """
        smooth_mask = ndi.gaussian_filter(nodule_mask.astype(np.float32), sigma=sigma)
        smooth_mask = np.clip(smooth_mask, 0.0, 1.0)
        blend = ct_patch*(1-alpha*smooth_mask) + nodule_patch*(alpha*smooth_mask)
        return blend


    def get_location (self, ct_array: np.array, lung_mask: np.array, region: str = None):
        """
        Select a 3D anatomical location within the lung mask for nodule insertion.
        This function segments the lung into clinically defined zones—upper, mid, lower—on both sides
        and returns representative coordinates based on clinical priors. For `region="stats"`, it samples
        a location based on probability-weighted likelihood of missed nodules.

        Parameters:
        - ct_array (np.ndarray): Original CT scan volume.
        - lung_mask (np.ndarray): Corresponding binary lung mask.
        - region (str, optional): Region name to return a specific location.
                                If "stats", a random region is selected based on predefined probabilities.
                                If None, all centers are returned.
        Returns:
        - tuple[int, int, int]: (z, y, x) location if a specific region is requested.
        - dict[str, tuple[int, int, int]]: All region locations if `region` is None.
        """

        num_slices = ct_array.shape[0]
        # lung slices are starting from 0.1n and ending at 0.8n
        z_position_ranges= {
            "upper": [int(0.1*num_slices), int(0.27*num_slices) ],
            "mid": [int(0.27*num_slices), int(0.63*num_slices) ],
            "lower": [int(0.63*num_slices), int(0.8*num_slices) ]
        }

        centers = {
            "right_apical": None,
            "right_mid_ventral": None,
            "right_hilar": None,
            "right_mid_lateral": None,
            "right_mid_dorsal": None,
            "right_basal": None,
            "left_apical": None,
            "left_mid_ventral": None,
            "left_hilar": None,
            "left_mid_lateral": None,
            "left_mid_dorsal": None,
            "left_basal": None
        }

    # # right apical
    #     submask = lung_mask[z_position_ranges["upper"][0]: z_position_ranges["upper"][1], :, : lung_mask.shape[2] // 2]
    #     #print(submask.shape)
    #     coords = (submask == 1).nonzero()
    #     rand_idx = np.random.choice(len(coords))
    #     #print("coords shape:", coords.shape)
    #     #print("coords[rand_idx]:", coords[rand_idx])
    #     z_sub, y, x = coords[rand_idx]
    #     z = z_sub + z_position_ranges["upper"][0]
    #     centers["right_apical"] = z, y, x

    # # left apical
    #     submask = lung_mask[z_position_ranges["upper"][0]: z_position_ranges["upper"][1], :, lung_mask.shape[2] // 2 :]
    #     coords = (submask == 1).nonzero()
    #     rand_idx = np.random.choice(len(coords))
    #     z_sub, y, x_sub = coords[rand_idx]
    #     z = z_sub + z_position_ranges["upper"][0]
    #     x = x_sub + lung_mask.shape[2] // 2
    #     centers["left_apical"] = z, y, x

    # # right basal
    #     submask = lung_mask[z_position_ranges["lower"][0]: z_position_ranges["lower"][1], :, : lung_mask.shape[2] // 2]
    #     coords = (submask == 1).nonzero()
    #     rand_idx = np.random.choice(len(coords))
    #     z_sub, y, x = coords[rand_idx]
    #     z = z_sub + z_position_ranges["lower"][0]
    #     centers["right_basal"] = z, y, x

    # # left basal
    #     submask = lung_mask[z_position_ranges["lower"][0]: z_position_ranges["lower"][1], :, lung_mask.shape[2] // 2 :]
    #     coords = (submask == 1).nonzero()
    #     rand_idx = np.random.choice(len(coords))
    #     z_sub, y, x_sub = coords[rand_idx]
    #     z = z_sub + z_position_ranges["lower"][0]
    #     x = x_sub + lung_mask.shape[2] // 2
    #     centers["left_basal"] = z, y, x


    # right mid_ventral
        submask = lung_mask[z_position_ranges["mid"][0]: z_position_ranges["mid"][1], :, : lung_mask.shape[2] // 2]
        coords = (submask == 1).nonzero()
        rand_idx = np.random.choice(len(coords))
        z = coords[rand_idx][0]
        submask_z = submask[z, :, :]
        ys, xs = np.where(submask_z == 1)
        y_min, y_max = ys.min(), ys.max()
        y_thresh = y_min + (y_max - y_min + 1) // 3
        top_third_indices = np.where(ys < y_thresh)[0]
        idx = np.random.choice(top_third_indices)
        z = z + z_position_ranges["mid"][0]
        y, x = ys[idx], xs[idx]
        centers["right_mid_ventral"] = z, y, x


    # left mid_ventral
        submask = lung_mask[z_position_ranges["mid"][0]: z_position_ranges["mid"][1], :, lung_mask.shape[2] // 2 :]
        coords = (submask == 1).nonzero()
        rand_idx = np.random.choice(len(coords))
        z = coords[rand_idx][0]
        submask_z = submask[z, :, :]
        ys, xs = np.where(submask_z == 1)
        y_min, y_max = ys.min(), ys.max()
        y_thresh = y_min + (y_max - y_min + 1) // 3
        top_third_indices = np.where(ys < y_thresh)[0]
        idx = np.random.choice(top_third_indices)
        z = z + z_position_ranges["mid"][0]
        y, x = ys[idx], xs[idx] + lung_mask.shape[2] // 2
        centers["left_mid_ventral"] = z, y, x


    #right mid_dorsal
        submask = lung_mask[z_position_ranges["mid"][0]: z_position_ranges["mid"][1], :, : lung_mask.shape[2] // 2]
        coords = (submask == 1).nonzero()
        rand_idx = np.random.choice(len(coords))
        z = coords[rand_idx][0]
        submask_z = submask[z, :, :]
        ys, xs = np.where(submask_z == 1)
        y_min, y_max = ys.min(), ys.max()
        y_thresh = y_max - (y_max - y_min + 1) // 3
        bottom_third_indices = np.where(ys > y_thresh)[0]
        idx = np.random.choice(bottom_third_indices)
        z = z + z_position_ranges["mid"][0]
        y, x = ys[idx], xs[idx]
        centers["right_mid_dorsal"] = z, y, x


    #left mid_dorsal
        submask = lung_mask[z_position_ranges["mid"][0]: z_position_ranges["mid"][1], :, lung_mask.shape[2] // 2 :]
        coords = (submask == 1).nonzero()
        rand_idx = np.random.choice(len(coords))
        z = coords[rand_idx][0]
        submask_z = submask[z, :, :]
        ys, xs = np.where(submask_z == 1)
        y_min, y_max = ys.min(), ys.max()
        y_thresh = y_max - (y_max - y_min + 1) // 3
        bottom_third_indices = np.where(ys > y_thresh)[0]
        idx = np.random.choice(bottom_third_indices)
        z = z + z_position_ranges["mid"][0]
        y, x = ys[idx], xs[idx] + lung_mask.shape[2] // 2
        centers["left_mid_dorsal"] = z, y, x

    #right mid_hilar
        submask = lung_mask[z_position_ranges["mid"][0]: z_position_ranges["mid"][1], :, : lung_mask.shape[2] // 2]
        coords = (submask == 1).nonzero()
        rand_idx = np.random.choice(len(coords))
        z = coords[rand_idx][0]
        submask_z = submask[z, :, :]
        ys, xs = np.where(submask_z == 1)
        y_min, y_max = ys.min(), ys.max()
        y_thresh1 = y_min + (y_max - y_min + 1) // 3
        y_thresh2 = y_max - (y_max - y_min + 1) // 3
        x_min, x_max = xs.min(), xs.max()
        x_thresh = x_min + (x_max - x_min + 1) // 2
        valid_indices = np.where((ys > y_thresh1) & (ys < y_thresh2) & (xs > x_thresh))[0]
        idx = np.random.choice(valid_indices)
        z = z + z_position_ranges["mid"][0]
        y, x = ys[idx], xs[idx]
        centers["right_hilar"] = z, y, x

    #left mid_hilar
        submask = lung_mask[z_position_ranges["mid"][0]: z_position_ranges["mid"][1], :, lung_mask.shape[2] // 2: ]
        coords = (submask == 1).nonzero()
        rand_idx = np.random.choice(len(coords))
        z = coords[rand_idx][0]
        submask_z = submask[z, :, :]
        ys, xs = np.where(submask_z == 1)
        y_min, y_max = ys.min(), ys.max()
        y_thresh1 = y_min + (y_max - y_min + 1) // 3
        y_thresh2 = y_max - (y_max - y_min + 1) // 3
        x_min, x_max = xs.min(), xs.max()
        x_thresh = x_min + (x_max - x_min + 1) // 2
        valid_indices = np.where((ys > y_thresh1) & (ys < y_thresh2) & (xs < x_thresh))[0]
        idx = np.random.choice(valid_indices)
        z = z + z_position_ranges["mid"][0]
        y, x = ys[idx], xs[idx] + lung_mask.shape[2] // 2
        centers["left_hilar"] = z, y, x

    # right mid_lateral
        submask = lung_mask[z_position_ranges["mid"][0]: z_position_ranges["mid"][1], :, : lung_mask.shape[2] // 2]
        coords = (submask == 1).nonzero()
        rand_idx = np.random.choice(len(coords))
        z = coords[rand_idx][0]
        submask_z = submask[z, :, :]
        ys, xs = np.where(submask_z == 1)
        y_min, y_max = ys.min(), ys.max()
        y_thresh1 = y_min + (y_max - y_min + 1) // 3
        y_thresh2 = y_max - (y_max - y_min + 1) // 3
        x_min, x_max = xs.min(), xs.max()
        x_thresh = x_min + (x_max - x_min + 1) // 2
        valid_indices = np.where((ys > y_thresh1) & (ys < y_thresh2) & (xs < x_thresh))[0]
        idx = np.random.choice(valid_indices)
        z = z + z_position_ranges["mid"][0]
        y, x = ys[idx], xs[idx]
        centers["right_mid_lateral"] = z, y, x

    # left mid_lateral
        submask = lung_mask[z_position_ranges["mid"][0]: z_position_ranges["mid"][1], :, lung_mask.shape[2] // 2: ]
        coords = (submask == 1).nonzero()
        rand_idx = np.random.choice(len(coords))
        z = coords[rand_idx][0]
        submask_z = submask[z, :, :]
        ys, xs = np.where(submask_z == 1)
        y_min, y_max = ys.min(), ys.max()
        y_thresh1 = y_min + (y_max - y_min + 1) // 3
        y_thresh2 = y_max - (y_max - y_min + 1) // 3
        x_min, x_max = xs.min(), xs.max()
        x_thresh = x_min + (x_max - x_min + 1) // 2
        valid_indices = np.where((ys > y_thresh1) & (ys < y_thresh2) & (xs > x_thresh))[0]
        idx = np.random.choice(valid_indices)
        z = z + z_position_ranges["mid"][0]
        y, x = ys[idx], xs[idx] + lung_mask.shape[2] // 2
        centers["left_mid_lateral"] = z, y, x


        probs= {
            "right_apical": 0.136,
            "right_mid_ventral": 0.142,
            "right_hilar": 0.046,
            "right_mid_lateral": 0.093,
            "right_mid_dorsal": 0.110,
            "right_basal": 0.070,
            "left_apical": 0.077,
            "left_mid_ventral": 0.049,
            "left_hilar": 0.044,
            "left_mid_lateral": 0.102,
            "left_mid_dorsal": 0.048,
            "left_basal": 0.083
        }

        positions = list(probs.keys())
        weights = list(probs.values())

        centers = {
        k: tuple(int(vv) for vv in v)
        for k, v in centers.items()
        }

        self.merge_centers= centers

        if region is not None:
            if region == "stats":
                selected_region = random.choices(positions, weights=weights, k=1)[0]
                return centers[selected_region]
            else:

                return centers[region]


        return centers


    def visualize_fake_nodule_region(self, ct_array: np.array, location: tuple, size: tuple = (100, 100, 100)):
        '''
        Extract a 3D region around the given location (FAKE NODULE) in the CT volume for visualization.
        '''
        zc, yc, xc = location
        dz, dy, dx = size[0] // 2, size[1] // 2, size[2] // 2

        z_start = max(zc - dz, 0)
        z_end = min(zc + dz + 1, ct_array.shape[0])
        y_start = max(yc - dy, 0)
        y_end = min(yc + dy + 1, ct_array.shape[1])
        x_start = max(xc - dx, 0)
        x_end = min(xc + dx + 1, ct_array.shape[2])

        patch = ct_array[z_start:z_end, y_start:y_end, x_start:x_end]

        return patch


    def visualize_real_nodule_region(self, ct_array: np.array, nodule_bbox: np.array, size: tuple = (100, 100, 100), index: int = 0):
        '''
        Extract a 3D region around the given location (REAL NODULE) in the CT volume for visualization.
        '''
        zn, yn, xn = nodule_bbox.shape
        start = False
        centers= []

        for z in range(zn):
            ys,xs = np.where(nodule_bbox[z] == 1)
            if len(xs)==0:
                start = False
            else:
                if start:
                    continue
                else:
                    start = True
                    xc, yc = xs.mean(), ys.mean()
                    centers.append((z, yc, xc))

        center = centers[index]
        center = int(center[0]), int(center[1]), int(center[2])
        nod = self.visualize_fake_nodule_region(ct_array= ct_array, location=center, size = size)
        return nod

    def ct_pipeline(self, nodule: Nodule(), region: str, size = 31, alpha: float = 0.8):

        '''
        MASTER FUNCTION FOR MERGING NODULE INTO CT.
        '''

        self.nodule = nodule
        self.region = region
        self.nodule_size = size
        self.blend_factor = alpha
        nodule_mat = nodule.nodule_mat

        self.load_ct(self.path)
        # centers = self.get_location(self.scans, self.masks)
        centers = {
            "right_apical": None,
            "right_mid_ventral": None,
            "right_hilar": None,
            "right_mid_lateral": None,
            "right_mid_dorsal": None,
            "right_basal": None,
            "left_apical": None,
            "left_mid_ventral": (151, 250, 150),
            "left_hilar": None,
            "left_mid_lateral": None,
            "left_mid_dorsal": None,
            "left_basal": None
        }
        merged = self.merge_nodule_with_ct(self.scans, nodule_mat, centers[region],size = size, alpha= alpha)
        merged_nod = self.visualize_fake_nodule_region(merged[0], centers[region], size= (100, 100, 100))
        #existing_nod = self.visualize_real_nodule_region(merged[0], self.bboxes_masks['nodule'][0], index= 2)
        zoom1 = [t / s for t, s in zip((128, 128, 128), merged_nod.shape)]
        zoom2 = [t / s for t, s in zip((128, 128, 128), merged_nod.shape)]
        merged_nod = zoom(merged_nod, zoom1, order= 1)
        # existing_nod = zoom(existing_nod, zoom2, order= 1)
        existing_nod = merged_nod
        print(f'merged nodule is at: {centers[region]}')

        self.merged_ct = merged[0]

        vis(merged[0], size= 8, title= "CT Scan")
        vis(self.masks, size= 8, title = "Lung Mask")
        vis(merged_nod, size= 8, title= "fake nodule")





g_nod, size= 8, title= "real nodule")    
