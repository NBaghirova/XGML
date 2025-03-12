import numpy as np
import os
import nibabel as nib
from sklearn.neighbors import KernelDensity
from KDEpy import FFTKDE
from dtw import dtw
import csv

# Define the root directory where the FDG-PET data is stored
# Replace this with the path to your own FDG-PET data directory
root_dir = "<path_to_pet_data>"  # Example: "/path/to/PET"

# Define the file pattern to look for in the data directory
file_suffix = "Warped.nii.gz"  # Files ending with this suffix will be processed

# Define the path to the brain atlas file
# Replace this with the path to your brain atlas file (e.g., Schaefer atlas)
path_to_atlas = "<path_to_brain_atlas>"  

# Use os.walk to traverse directories and find files that match the pattern
for dirpath, _, filenames in os.walk(root_dir):
    # List to store all the file paths
    file_paths = [os.path.join(dirpath, filename) for filename in filenames if filename.endswith(file_suffix)]

    if not file_paths:
        continue  # Skip directories without matching files

    # Extract the patient directory from the path
    patient_dir = os.path.basename(dirpath)

    # Check if a CSV file already exists for this patient
    csv_file_path = os.path.join(dirpath, f"{patient_dir}_similarity.csv")
    if os.path.exists(csv_file_path):
        print(f"Similarities for {patient_dir} already exist. Skipping this subject.")
        continue

    # Load the atlas
    try:
        atlas = nib.load(path_to_atlas)
    except FileNotFoundError:
        raise FileNotFoundError("Brain atlas file not found. Please provide a valid path to the brain atlas.")

    atlas_data = atlas.get_fdata()

    # Initialize lists for storing similarity results
    similarity_brain_regions = []

    # Iterate over each image file in the current patient directory
    for path in file_paths:
        try:
            # Import data from the image
            image = nib.load(path)
        except nib.filebasedimages.ImageFileError as e:
            print(f"Skipping invalid file: {path} ({e})")
            continue

        image_data = image.get_fdata()

        # Print shapes for debugging
        print(f"Processing file: {path}")
        print(f"Atlas shape: {atlas_data.shape}, Image shape: {image_data.shape}")

        # Check if the shapes of atlas_data and image_data match
        if atlas_data.shape != image_data.shape:
            print(f"Shape mismatch: atlas {atlas_data.shape}, image {image_data.shape}. Skipping file.")
            continue

        # Create a list of brain regions and remove the region corresponding to 0
        brain_regions = np.unique(atlas_data)
        brain_regions = brain_regions[brain_regions != 0]

        # Create an empty dictionary with regions as keys
        region_values = {region: [] for region in brain_regions}

        # Assign values to the brain regions
        for region in brain_regions:
            region_coords = np.where(atlas_data == region)

            # Ensure that region_coords are within the bounds of image_data
            if np.any(region_coords[0] >= image_data.shape[0]) or \
               np.any(region_coords[1] >= image_data.shape[1]) or \
               np.any(region_coords[2] >= image_data.shape[2]):
                print(f"Skipping region {region} due to coordinate out of bounds")
                continue

            values = image_data[region_coords]
            region_values[region].extend(values)

        # Convert lists to numpy arrays in the dictionary
        region_values = {region: np.array(values) for region, values in region_values.items()}

        # Apply Kernel Density Estimation
        region_values_KDE = {}
        region_values_KDE_densities = {}
        x_range = {}

        for region, values in region_values.items():
            if len(values) == 0:
                continue  # Skip regions with no values

            reshaped_values = values.reshape(-1, 1)  # Correct shape for 1D KDE
            x_d = np.linspace(np.min(values) - 0.3, np.max(values) + 0.3, num=2**10).reshape(-1, 1)
            log_density = FFTKDE(kernel='gaussian', bw='ISJ').fit(reshaped_values, weights=None).evaluate(x_d)

            # Store the density values
            region_values_KDE_densities[region] = log_density
            x_range[region] = x_d

        # Compute similarity using DTW
        for i, region1 in enumerate(brain_regions):
            for j, region2 in enumerate(brain_regions):
                if i < j:  # Avoid calculating similarity more than once for the same regions
                    alignment = dtw(region_values_KDE_densities.get(region1, []),
                                    region_values_KDE_densities.get(region2, []),
                                    dist_method='euclidean', keep_internals=True)
                    distance = alignment.distance
                    new_tuple = (region1, region2, distance)
                    similarity_brain_regions.append(new_tuple)

    # Define the output file path
    output_file = os.path.join(dirpath, f"{patient_dir}_similarity.csv")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write the similarity results to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Region 1", "Region 2", "Similarity"])
        writer.writerows(similarity_brain_regions)

    print(f"Similarities for {patient_dir} have been written to {output_file}")
