# degirum-face
DeGirum Face Recognition Package

## Installation

`pip install degirum-face`

## Running GUI Example

```
git clone https://github.com/DeGirum/degirum_face
cd degirum_face/examples
python3 ./face_tracking_web_app.py`
```

Then open browser and navigate to http://localhost:8080

## Examples

| File | Description |
|------|-------------|
|Tutorials.ipynb| Jupyter notebook with face recognition & tracking tutorials. |
|face_recognition_enroll.py| Example how to add face embeddings to the ReID database from images.|
|face_recognition_simple.py| Example how to recognize people from a set of images. Requires face database to be filled by `face_recognition_enroll.py`.|
|face_tracking_web_app.py| Full-featured web application for real time intruder detection with NVR capabilities and notifications. Has GUI for adding known faces to face database based on captured video clips of unknown faces. |
|face_tracking_simple.py| Subset of `face_tracking_web_app.py`, which does real time intruder detection part displaying live preview in local window. Requires known face database to be filled by `face_tracking_web_app.py` or `face_tracking_add_embeddings.py` |
|face_tracking_add_embeddings.py| Subset of `face_tracking_web_app.py`, which adds known faces to face database based on captured video clips of unknown faces. Requires such video clips to be captured by either `face_tracking_web_app.py` or `face_tracking_simple.py` |
