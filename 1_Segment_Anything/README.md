```python
python3 1_extract_frames.py celadon_vessel.mp4 extracted_frames -r 5
```


```python
python3 2_resize.py extracted_frames --max-dimension 1000
```

```python
python3 3_segmenter.py
```







```python
python 5_split_rename.py --input extracted_objects --output celadon_vessel --test 15
```

python3 run_video.py --video-path celadon_vessel.mp4



python3 run_glomap.py --image_path db_split


python3 3_split_rename.py --input "extracted_frames" --masks "extracted_frames/segmentation_output/masks" --output "db_split"