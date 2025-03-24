# Create a project directory
mkdir -p db/colmap/sparse

# Feature extraction
colmap feature_extractor \
    --database_path db/colmap/database.db \
    --image_path db/images

# Feature matching
colmap exhaustive_matcher \
    --database_path db/colmap/database.db

# Sparse reconstruction
colmap mapper \
    --database_path db/colmap/database.db \
    --image_path db/images \
    --output_path db/colmap/sparse