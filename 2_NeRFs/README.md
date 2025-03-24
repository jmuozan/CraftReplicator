- Why scene4 reconstruction folder has nothing and the rest do?

- Mask out images after reconstructions but before combining reconstructions. 

- generate transforms and try to train 






















```bash
python3 1_colmap2nerf.py --run_colmap --images tests_colmap/test_1 --colmap_matcher exhaustive --colmap_camera_model SIMPLE_PINHOLE
```

```bash
python3 1_colmap2nerf.py --images tests_colmap/test_1 --colmap_matcher exhaustive --run_colmap --aabb_scale 32
```




## COLMAP

```bash
# Create necessary directories
mkdir -p output/extracted_objects/sparse
mkdir -p output/extracted_objects/sparse_text

colmap feature_extractor \
  --image_path /Users/jorgemuyo/Desktop/CraftReplicator/1_Segment_Anything/extracted_objects \
  --database_path database.db \
  --ImageReader.default_focal_length_factor 1.2 \
  --SiftExtraction.max_num_features 8192 \
  --SiftExtraction.first_octave -1

# Match features
colmap exhaustive_matcher \
    --database_path database.db

# Run GLOMAP mapper
glomap mapper \
  --database_path database.db \
  --image_path /Users/jorgemuyo/Desktop/CraftReplicator/1_Segment_Anything/extracted_objects \
  --output_path output/extracted_objects/sparse \
  --RelPoseEstimation.max_epipolar_error 10

colmap model_converter \
    --input_path output/extracted_objects/sparse/0 \
    --output_path output/extracted_objects/sparse_text \
    --output_type TXT

python 1_colmap2nerf.py \
    --images /Users/jorgemuyo/Desktop/CraftReplicator/1_Segment_Anything/extracted_objects \
    --text output/extracted_objects/sparse_text \
    --out transforms.json \
    --aabb_scale 16


```






```bash
# Create necessary directories
mkdir -p output/extracted_objects/sparse
mkdir -p output/extracted_objects/sparse_text

# Extract features
colmap feature_extractor \
    --image_path ./extracted_objects \
    --database_path ./database.db

# Match features
colmap exhaustive_matcher \
    --database_path ./database.db

# Run GLOMAP mapper
glomap mapper \
    --database_path ./database.db \
    --image_path ./extracted_objects \
    --output_path ./output/extracted_objects/sparse

# Convert model from binary to text format
colmap model_converter \
    --input_path ./output/extracted_objects/sparse/0 \
    --output_path ./output/extracted_objects/sparse_text \
    --output_type TXT

# Generate transforms.json for NeRF
python 1_colmap2nerf.py \
    --images ./extracted_objects \
    --text ./output/extracted_objects/sparse_text \
    --out ./transforms.json \
    --aabb_scale 16
```




mkdir -p output/extracted_objects/sparse
mkdir -p output/extracted_objects/sparse_text

# Extract features with shared camera parameters
colmap feature_extractor \
    --image_path ./extracted_objects \
    --database_path ./database.db \
    --ImageReader.single_camera 1 \
    --SiftExtraction.max_num_features 8192

# Match features with increased epipolar error tolerance
colmap exhaustive_matcher \
    --database_path ./database.db \
    --SiftMatching.guided_matching 1

# Run GLOMAP mapper with optimized settings
glomap mapper \
    --database_path ./database.db \
    --image_path ./extracted_objects \
    --output_path ./output/extracted_objects/sparse \
    --RelPoseEstimation.max_epipolar_error 4.0 \
    --TrackEstablishment.max_num_tracks 163000

# Convert and generate transforms.json as before
mkdir -p output/extracted_objects/sparse_text
colmap model_converter \
    --input_path ./output/extracted_objects/sparse/0 \
    --output_path ./output/extracted_objects/sparse_text \
    --output_type TXT

python 1_colmap2nerf.py \
    --images ./extracted_objects \
    --text ./output/extracted_objects/sparse_text \
    --out ./transforms.json \
    --aabb_scale 16