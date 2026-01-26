import sys
import os

# Add the folder containing defacer.py to the search path
sys.path.append(os.path.join(os.getcwd(), "model_distribution"))

from defacer import Defacer


# 1. Define paths
# Point this to the FOLDER containing the .dcm files for ONE exam
dicom_input_dir = "./test_exam" 
output_dir = "./anonymized_output2"
verification_dir = "./verification_images"

# Create output folders if they don't exist
if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(verification_dir): os.makedirs(verification_dir)

# 2. Initialize the tool
# This loads the model weights from the /model folder automatically
print("Loading model...")
coder = Defacer()

# 3. Choose features to remove
# (Eyes, Nose, Ears, Mouth) -> 1 to remove, 0 to keep
deid_mask = (1, 1, 1, 1)

# 4. Run the process
print("Starting de-identification...")
result = coder.Deidentification_image_dcm(
    where=deid_mask,
    dicom_path=dicom_input_dir,
    dest_path=output_dir,
    verif_path=verification_dir,
    url="",           # Only used for web-services, leave empty
    prefix="DeID_"    # Prefix added to the output filenames
)

if result["success"]:
    print(f"Success! Files saved to: {result['path']}")
else:
    print(f"Error: {result.get('msg')}")