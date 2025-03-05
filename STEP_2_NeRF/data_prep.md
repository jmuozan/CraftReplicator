```bash
data/
  └── your_scene/
      ├── images/
      │   ├── image_001.png
      │   ├── image_002.png
      │   └── ...
      └── transforms_train.json
```

```bash
mkdir -p data/porcelain/sparse

colmap mapper \
    --database_path data/porcelain/database.db \
    --image_path data/porcelain \
    --output_path data/porcelain/sparse \
    --Mapper.filter_max_reproj_error 4 \
    --Mapper.tri_min_angle 4 \
    --Mapper.min_num_matches 15
```

```bash
python colmap_to_nerf_fixed.py --colmap_dir data/porcelain/sparse/0 --output_dir data/porcelain_nerf --image_dir data/porcelain --square
```

```bash
ls -la data/porcelain_nerf/images/ | wc -l
```

```bash
python train.py \
  --dataset_name blender \
  --root_dir data/porcelain_nerf \
  --N_importance 64 \
  --img_wh 1080 1080 \
  --noise_std 0 \
  --num_epochs 20 \
  --batch_size 1024 \
  --optimizer adam \
  --lr 5e-4 \
  --lr_scheduler cosine \
  --exp_name porcelain_exp \
  --data_perturb color occ \
  --encode_t \
  --encode_a \
  --beta_min 0.1
```