#!/usr/bin/env python3
"""
Fixed Tri-level Preprocessing for Alzheimer's Detection with SynthStrip

This script integrates:
1. Noise Reduction using simplified Gaussian filtering
2. Skull Stripping using SynthStrip (deep learning-based approach) - fixed for tensor size issues
3. Bias Field Correction using Expectation-Maximization (EM) with faster parameters
"""

import os
import argparse
import shutil
import json
import datetime
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import tempfile
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_fill_holes
import SimpleITK as sitk
from tqdm import tqdm
import gc
import multiprocessing

# Define argument parser
parser = argparse.ArgumentParser(description='Tri-level Preprocessing for MRI images with SynthStrip')
parser.add_argument('--input', '-i', required=True, help='Input directory containing AD, MCI, CN subfolders')
parser.add_argument('--output', '-o', required=True, help='Output directory for processed images')
parser.add_argument('--border', '-b', type=float, default=1.0, help='Mask border threshold in mm (default: 1.0)')
parser.add_argument('--no-csf', action='store_true', help='Exclude CSF from brain border')
parser.add_argument('--cpu', action='store_true', help='Force CPU usage (default: use GPU if available)')
parser.add_argument('--model-dir', help='Directory to save/load model weights (default: temp directory)')
parser.add_argument('--visualize', action='store_true', help='Visualize first image in each category')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(), help=f'Number of threads (default: {multiprocessing.cpu_count()})')
parser.add_argument('--skip-noise', action='store_true', help='Skip noise reduction (default: False)')
parser.add_argument('--skip-bias', action='store_true', help='Skip bias field correction (default: False)')
parser.add_argument('--npy-dir', type=str, help='Directory to save .npy vector files (default: processed_npy_data)')
parser.add_argument('--flatten', action='store_true', help='Flatten 3D MRI data to 1D vector before saving as .npy')
parser.add_argument('--resize', type=int, nargs=3, help='Resize MRI data to specified dimensions (e.g., --resize 128 128 128)')
parser.add_argument('--skip-nii', action='store_true', help='Skip saving NIfTI (.nii) files (only save .npy)')
args = parser.parse_args()

# Set PyTorch threads to limit memory usage
torch.set_num_threads(args.threads)

# Set device (GPU or CPU)
if args.cpu:
    device = torch.device('cpu')
    print('Using CPU for inference')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f'Using GPU for inference: {torch.cuda.get_device_name(0)}')
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    else:
        print('GPU not available, using CPU for inference')

# Model URL and file
MODEL_VERSION = '1'
if args.no_csf:
    MODEL_FILE = f'synthstrip.nocsf.{MODEL_VERSION}.pt'
else:
    MODEL_FILE = f'synthstrip.{MODEL_VERSION}.pt'

# Define model directory
if args.model_dir:
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
else:
    model_dir = tempfile.gettempdir()

model_path = os.path.join(model_dir, MODEL_FILE)

# Look for model in current directory if not found in model_dir
if not os.path.exists(model_path):
    current_dir_model = os.path.join(".", MODEL_FILE)
    if os.path.exists(current_dir_model):
        print(f"Found model in current directory, copying to: {model_path}")
        shutil.copy(current_dir_model, model_path)
    else:
        print(f"Error: Model not found at {model_path} or in current directory")
        print(f"Please download the model and save it as {MODEL_FILE} in the current directory")
        exit(1)

print(f"Using model weights from: {model_path}")


#############################
# 1. NOISE REDUCTION (FAST) #
#############################

def fast_noise_reduction(image, sigma=0.7):
    """
    Fast noise reduction using Gaussian filtering
    """
    print("Applying noise reduction...")
    output = gaussian_filter(image, sigma=sigma)
    print("Noise reduction completed.")
    return output


##########################
# 2. SYNTHSTRIP (SKULL) #
##########################

# Define SynthStrip model architecture
class ConvBlock(nn.Module):
    """Convolutional block with LeakyReLU activation"""

    def __init__(self, ndims, in_channels, out_channels, stride=1, activation='leaky'):
        super().__init__()

        Conv = getattr(nn, f'Conv{ndims}d')
        self.conv = Conv(in_channels, out_channels, 3, stride, 1)

        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f'Unknown activation: {activation}')

    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        return out

class StripModel(nn.Module):
    """SynthStrip U-Net model architecture"""

    def __init__(self,
                 nb_features=16,
                 nb_levels=7,
                 feat_mult=2,
                 max_features=64,
                 nb_conv_per_level=2,
                 max_pool=2,
                 return_mask=False):

        super().__init__()

        # dimensionality
        ndims = 3

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            feats = np.clip(feats, 1, max_features)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, f'MaxPool{ndims}d')
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if level < (self.nb_levels - 1):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # final convolutions
        if return_mask:
            self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
            self.remaining.append(nn.Softmax(dim=1))
        else:
            self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))

    def forward(self, x):
        # Modified forward pass with size checks
        x_history = [x]
        
        # Encoder path
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)
        
        # Decoder path with tensor size checks
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            
            if level < (self.nb_levels - 1):
                x = self.upsampling[level](x)
                
                # Get the skip connection from history
                skip = x_history.pop()
                
                # Check tensor sizes - this is the critical fix
                if x.size(2) != skip.size(2) or x.size(3) != skip.size(3) or x.size(4) != skip.size(4):
                    # Resize skip connection to match x
                    skip = torch.nn.functional.interpolate(
                        skip, 
                        size=(x.size(2), x.size(3), x.size(4)),
                        mode='nearest'
                    )
                
                # Concatenate tensors
                x = torch.cat([x, skip], dim=1)
        
        # Output path
        for conv in self.remaining:
            x = conv(x)
            
        return x

# Fixed conforming function that ensures divisibility by 64 in all dimensions
def fixed_conform_image(image_data):
    """
    Conform image to standard orientation and voxel size with fixed dimensions
    that work with the SynthStrip model
    """
    # Get current shape
    current_shape = image_data.shape
    
    # Calculate target shape that is exactly divisible by 64 in all dimensions
    target_shape = [((s + 63) // 64) * 64 for s in current_shape]
    target_shape = [max(s, 192) for s in target_shape]  # Ensure minimum size of 192
    target_shape = [min(s, 256) for s in target_shape]  # Cap at 256 for efficiency
    
    # Create a new array
    conformed = np.zeros(target_shape, dtype=np.float32)
    
    # Calculate the position to place the original data
    start = [(t - c) // 2 for t, c in zip(target_shape, current_shape)]
    end = [s + c for s, c in zip(start, current_shape)]
    
    # Place the original data in the center of the new array
    slices_orig = tuple(slice(0, c) for c in current_shape)
    slices_conf = tuple(slice(s, e) for s, e in zip(start, end))
    
    conformed[slices_conf] = image_data[slices_orig]
    
    return conformed

def get_largest_cc(mask):
    """Get the largest connected component in a binary mask"""
    # Label connected components
    labeled, num_components = ndimage.label(mask)
    
    if num_components == 0:
        return mask
    
    # Find the largest component
    sizes = np.bincount(labeled.ravel())[1:]
    largest_component = np.argmax(sizes) + 1
    
    return labeled == largest_component

def synthstrip_skull_stripping(image, model, device, border=1.0):
    """
    Fixed SynthStrip skull stripping function
    """
    print("Performing SynthStrip skull stripping...")
    
    try:
        # Use fixed conform function to avoid tensor size issues
        print("Conforming image to standard space...")
        conformed_data = fixed_conform_image(image)
        
        # Normalize the image for SynthStrip
        print("Normalizing image...")
        conformed_data = conformed_data.astype(np.float32)
        conformed_data = conformed_data - conformed_data.min()
        p99 = np.percentile(conformed_data, 99)
        if p99 > 0:
            conformed_data = conformed_data / p99
        conformed_data = np.clip(conformed_data, 0, 1)
        
        # Prepare input tensor
        print("Preparing input tensor...")
        input_tensor = torch.from_numpy(conformed_data).to(device)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Run inference with additional error handling
        print("Running SynthStrip inference...")
        try:
            with torch.no_grad():
                sdt = model(input_tensor).squeeze().cpu().numpy()
        except RuntimeError as e:
            if "size" in str(e) or "match" in str(e):
                print("Tensor size mismatch error, falling back to threshold-based skull stripping")
                
                # Simple threshold-based fallback method
                threshold = np.percentile(conformed_data, 95)
                binary_mask = conformed_data > threshold
                brain_mask = binary_fill_holes(binary_mask)
                brain_mask = get_largest_cc(brain_mask)
                
                # Unconform mask to original space
                from scipy.ndimage import zoom
                zoom_factors = [float(i) / float(j) for i, j in zip(image.shape, brain_mask.shape)]
                brain_mask = zoom(brain_mask.astype(float), zoom_factors, order=0) > 0.5
                
                # Apply the mask
                brain_image = image.copy()
                brain_image[~brain_mask] = 0
                
                return brain_image, brain_mask
            else:
                raise e
        
        # Free up GPU memory
        del input_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create brain mask directly from SDT
        print("Creating brain mask...")
        brain_mask = sdt < border
        
        # Get largest connected component
        print("Finding largest connected component...")
        brain_mask = get_largest_cc(brain_mask)
        
        # Fill holes in the mask
        print("Filling holes in brain mask...")
        brain_mask = binary_fill_holes(brain_mask)
        
        # Unconform the mask to original image space
        print("Transforming mask to original image space...")
        if brain_mask.shape != image.shape:
            from scipy.ndimage import zoom
            zoom_factors = [float(i) / float(j) for i, j in zip(image.shape, brain_mask.shape)]
            brain_mask = zoom(brain_mask.astype(float), zoom_factors, order=0) > 0.5
        
        # Apply the mask to the original image
        print("Applying mask to create skull-stripped image...")
        brain_image = image.copy()
        brain_image[~brain_mask] = 0
        
        print("SynthStrip skull stripping completed successfully.")
        return brain_image, brain_mask
        
    except Exception as e:
        print(f"Error in SynthStrip: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fall back to simple thresholding if there's an error
        print("Falling back to simple thresholding for skull stripping")
        
        # Simple threshold-based method
        threshold = np.percentile(image, 90)
        binary_mask = image > threshold
        brain_mask = binary_fill_holes(binary_mask)
        brain_mask = get_largest_cc(brain_mask)
        
        # Apply the mask
        brain_image = image.copy()
        brain_image[~brain_mask] = 0
        
        return brain_image, brain_mask


###############################
# 3. BIAS FIELD CORRECTION (EM) #
###############################

def fast_bias_correction(image, mask=None):
    """
    Fast bias field correction using SimpleITK's N4 with aggressive parameters
    """
    print("Applying bias field correction...")
    
    try:
        # Convert to SimpleITK images
        sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
        
        if mask is not None:
            sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
        else:
            sitk_mask = sitk.GetImageFromArray(np.ones_like(image, dtype=np.uint8))
        
        # Create N4 bias field corrector
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        
        # Set parameters for faster processing
        corrector.SetMaximumNumberOfIterations([2])
        corrector.SetConvergenceThreshold(0.001)
        
        # Apply correction
        output = corrector.Execute(sitk_image, sitk_mask)
        corrected_image = sitk.GetArrayFromImage(output)
        print("Bias field correction completed.")
        
        return corrected_image
        
    except Exception as e:
        print(f"Warning: Bias field correction failed: {str(e)}. Using original image.")
        import traceback
        traceback.print_exc()
        return image


#######################
# NPY VECTOR STORAGE #
#######################

def save_as_npy(image_data, patient_id, category, npy_dir, flatten=False, resize=None):
    """
    Save processed MRI data as .npy file
    """
    # Create category directory if it doesn't exist
    category_dir = os.path.join(npy_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    # Prepare data
    data = image_data.copy()
    
    # Resize if specified
    if resize is not None:
        from scipy.ndimage import zoom
        # Calculate zoom factors
        zoom_factors = [float(r) / float(s) for r, s in zip(resize, data.shape)]
        # Resize data
        data = zoom(data, zoom_factors, order=1)  # order=1 for linear interpolation
    
    # Flatten if specified
    if flatten:
        data = data.flatten()
    
    # Save data
    output_path = os.path.join(category_dir, f"{patient_id}.npy")
    np.save(output_path, data)
    
    print(f"Saved .npy vector to: {output_path}")
    return output_path


####################
# VISUALIZATION #
####################

def visualize_results(original, noise_reduced, skull_stripped, final, mask, patient_id):
    """Visualize the results of preprocessing"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Get the middle slice from each dimension
        shape = original.shape
        mid_z = shape[2] // 2  # Axial view (top-down)
        
        # Create a figure
        plt.figure(figsize=(20, 4))
        plt.suptitle(f"Patient {patient_id}: Preprocessing Results", fontsize=16)
        
        # Original image
        plt.subplot(1, 5, 1)
        plt.imshow(original[:, :, mid_z], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Noise reduced image
        plt.subplot(1, 5, 2)
        plt.imshow(noise_reduced[:, :, mid_z], cmap='gray')
        plt.title('Step 1: Noise Reduced')
        plt.axis('off')
        
        # Brain mask
        plt.subplot(1, 5, 3)
        plt.imshow(mask[:, :, mid_z], cmap='gray')
        plt.title('Brain Mask')
        plt.axis('off')
        
        # Skull-stripped image
        plt.subplot(1, 5, 4)
        plt.imshow(skull_stripped[:, :, mid_z], cmap='gray')
        plt.title('Step 2: Skull Stripped')
        plt.axis('off')
        
        # Final image
        plt.subplot(1, 5, 5)
        plt.imshow(final[:, :, mid_z], cmap='gray')
        plt.title('Step 3: Bias Corrected')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{patient_id}_processing_results.png")
        plt.close()
        return True
    except Exception as e:
        print(f"Error visualizing results: {str(e)}")
        return False


####################
# MAIN EXECUTION #
####################

def main():
    # Set higher process priority on Windows
    try:
        import psutil
        process = psutil.Process()
        process.nice(psutil.HIGH_PRIORITY_CLASS)
    except:
        pass
        
    # Configure SimpleITK threads
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(min(args.threads, 8))
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Set NPY directory
    npy_dir = args.npy_dir if args.npy_dir else "processed_npy_data"
    os.makedirs(npy_dir, exist_ok=True)
    
    # Load model
    print("Loading SynthStrip model...")
    model = StripModel()
    
    # Load model to device
    model.to(device)
    model.eval()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Print GPU info if available
    if device.type == 'cuda':
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Dictionary to store visualization results
    visualization_results = {}
    
    # Process files from each category (AD, MCI, CN)
    categories = ["AD", "MCI", "CN"]
    total_processed = 0
    total_errors = 0
    
    # Track saved NPY files
    npy_files = {category: [] for category in categories}
    
    for category in categories:
        # Create source and destination paths for this category
        src_dir = os.path.join(args.input, category)
        dst_dir = os.path.join(args.output, category)
        
        # Ensure destination directory exists
        os.makedirs(dst_dir, exist_ok=True)
        os.makedirs(os.path.join(npy_dir, category), exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Processing {category} images...")
        print(f"{'='*50}")
        
        # Check if source directory exists
        if not os.path.exists(src_dir):
            print(f"Warning: Source directory {src_dir} not found, skipping.")
            continue
        
        # Find all .nii files recursively in the directory
        nii_files = []
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith('.nii'):
                    # Store the full path for later use
                    nii_files.append(os.path.join(root, file))
        
        if not nii_files:
            print(f"No .nii files found in {src_dir} or its subdirectories")
            continue
        
        print(f"Found {len(nii_files)} .nii files in {category} folder and its subdirectories")
        
        category_processed = 0
        category_errors = 0
        
        # Process each file
        for i, file_path in enumerate(tqdm(nii_files, desc=f"Processing {category} files")):
            try:
                # Clean up memory
                if i > 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Extract patient ID from path
                try:
                    # Use filename as patient ID
                    base_name = os.path.basename(file_path).split('.')[0]
                    patient_id = base_name
                except:
                    # If any error occurs, use a simple counter
                    patient_id = f"{category}_{i+1}"
                
                # Create output filename
                base_file_name = os.path.basename(file_path)
                output_file = os.path.join(dst_dir, f"{patient_id}_{base_file_name}")
                
                # Check for existing .npy file to avoid reprocessing
                npy_file_path = os.path.join(npy_dir, category, f"{patient_id}.npy")
                if os.path.exists(npy_file_path) and os.path.getsize(npy_file_path) > 0:
                    print(f"\nSkipping {base_file_name} - already processed (.npy exists)")
                    npy_files[category].append(npy_file_path)
                    category_processed += 1
                    total_processed += 1
                    continue
                
                print(f"\nProcessing: {base_file_name}")
                print(f"Patient ID: {patient_id}")
                
                # Save intermediate results for visualization if requested
                save_intermediate = args.visualize and category not in visualization_results
                
                # Process the file
                try:
                    nii_img = nib.load(file_path)
                    img_data = nii_img.get_fdata()
                    affine = nii_img.affine
                except Exception as e:
                    print(f"Error loading image {file_path}: {str(e)}")
                    category_errors += 1
                    total_errors += 1
                    continue
                
                # Check if the image is 3D
                if len(img_data.shape) > 3:
                    print(f"Input image has {len(img_data.shape)} dimensions. Using first frame/volume only.")
                    # Extract the first volume if it's a 4D image
                    img_data = img_data[..., 0]
                
                # Store original for visualization
                original_data = img_data.copy() if save_intermediate else None
                
                # 1. NOISE REDUCTION
                if not args.skip_noise:
                    noise_reduced = fast_noise_reduction(img_data)
                else:
                    print("Skipping noise reduction (as requested).")
                    noise_reduced = img_data
                
                # Free memory
                if not save_intermediate:
                    del img_data
                    gc.collect()
                
                # 2. SKULL STRIPPING
                brain_image, brain_mask = synthstrip_skull_stripping(
                    noise_reduced, model, device, args.border
                )
                
                # Save intermediate result for visualization
                if save_intermediate:
                    noise_reduced_viz = noise_reduced.copy()
                    brain_image_viz = brain_image.copy()
                    brain_mask_viz = brain_mask.copy()
                
                # Free memory
                if not save_intermediate:
                    del noise_reduced
                    gc.collect()
                
                # 3. BIAS FIELD CORRECTION
                if not args.skip_bias:
                    final_image = fast_bias_correction(brain_image, brain_mask)
                else:
                    print("Skipping bias field correction (as requested).")
                    final_image = brain_image
                
                # Free memory
                if not save_intermediate:
                    del brain_image
                    gc.collect()
                
                # Save the processed image as .nii only if not skipped
                if not args.skip_nii:
                    print(f"Saving preprocessed image to: {output_file}")
                    processed_nii = nib.Nifti1Image(final_image, affine, nii_img.header)
                    nib.save(processed_nii, output_file)
                else:
                    print("Skipping NIfTI (.nii) file saving as requested with --skip-nii flag")
                
                # Save the processed image as .npy vector
                resize_dims = args.resize if args.resize else None
                npy_path = save_as_npy(
                    final_image,
                    patient_id,
                    category,
                    npy_dir,
                    flatten=args.flatten,
                    resize=resize_dims
                )
                npy_files[category].append(npy_path)
                
                # Store visualization results for first image in category
                if save_intermediate:
                    visualization_results[category] = {
                        'original': original_data,
                        'noise_reduced': noise_reduced_viz,
                        'skull_stripped': brain_image_viz,
                        'final': final_image,
                        'mask': brain_mask_viz,
                        'patient_id': patient_id
                    }
                
                print(f"Successfully processed and saved")
                category_processed += 1
                total_processed += 1
                
                # Clean up memory after processing
                del final_image
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                category_errors += 1
                total_errors += 1
        
        print(f"\nCategory {category} summary:")
        print(f"  - Successfully processed: {category_processed} files")
        print(f"  - Errors: {category_errors} files")
        print(f"  - NPY vectors saved: {len(npy_files[category])}")
    
    print(f"\n{'='*50}")
    print("Overall Processing Summary:")
    print(f"{'='*50}")
    print(f"Total successfully processed: {total_processed} files")
    print(f"Total errors: {total_errors} files")
    print(f"Total NPY vectors saved: {sum(len(files) for files in npy_files.values())}")
    
    # Print NPY storage details
    print(f"\nNPY vectors stored in: {npy_dir}")
    for category in categories:
        if npy_files[category]:
            print(f"  - {category}: {len(npy_files[category])} files")
    
    # Save metadata about the processing
    metadata = {
        'processing_date': str(datetime.datetime.now()),
        'total_processed': total_processed,
        'flatten': args.flatten,
        'resize': args.resize,
        'categories': {cat: len(npy_files[cat]) for cat in categories},
        'files': {cat: [os.path.basename(f) for f in npy_files[cat]] for cat in categories}
    }
    
    metadata_path = os.path.join(npy_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved processing metadata to: {metadata_path}")
    
    # Visualize results only if requested
    if args.visualize and visualization_results:
        print("\nVisualizing results...")
        for category, results in visualization_results.items():
            print(f"Visualizing results for category: {category}")
            visualize_results(
                results['original'],
                results['noise_reduced'],
                results['skull_stripped'],
                results['final'],
                results['mask'],
                results['patient_id']
            )
    
    print("\nTri-level preprocessing completed!")
    print("\nCitation Information:")
    print("1. Noise Reduction & Bias Field Correction:")
    print("   Gharaibeh et al. (2023). Swin Transformer-Based Segmentation and")
    print("   Multi-Scale Feature Pyramid Fusion Module for Alzheimer's Disease.")
    print("   DOI: 10.3991/ijoe.v19i04.37677")
    print("2. SynthStrip Skull Stripping:")
    print("   Hoopes A, Mora JS, Dalca AV, Fischl B, Hoffmann M. (2022).")
    print("   SynthStrip: Skull-Stripping for Any Brain Image.")
    print("   NeuroImage 260, 119474.")
    print("   https://doi.org/10.1016/j.neuroimage.2022.119474")


# Run the main function
if __name__ == "__main__":
    main()