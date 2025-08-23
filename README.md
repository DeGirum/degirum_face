# DeGirum Face Recognition SDK

Unified, explicit, and modular face recognition pipeline for Python.

## Features

- **Explicit pipeline:** Each step (detection, alignment, embedding, identification) is clear and composable.
- **Multi-hardware support:** Works seamlessly with a variety of AI accelerators and CPUs (Hailo, CUDA, CPU, etc).
- **Database-backed:** Robust, deduplicated face database (LanceDB).
- **Flexible API:** Use high-level or stepwise face recognition, enrollment, and verification.
- **Customizable:** Swap models, configure thresholds, and control every stage.

## Installation

```bash
pip install degirum-face
```

## Pipeline Overview

The pipeline is explicit, modular, and robust. Each step is a method:

1. **Face Detection:** Locate faces in an image. Returns all detections, unfiltered.
2. **Face Filtering (Optional):** Annotate detections with `face_rejected` and `reject_reason` if a face is low quality. Filtering is annotation-based and does not remove detections.
3. **Face Alignment:** Crop and align faces for embedding. Skips detections with `face_rejected=True`.
4. **Face Embedding:** Generate feature vectors for each face.
5. **Identification:** Match embeddings to database and assign labels.

You can run the full pipeline in one call, or step by step for maximum control.

**Key Design Principles:**
- Filtering is annotation-based: detections are never removed, only marked as rejected.
- All pipeline steps always receive the full list of detections and are robust to missing or repeated processing.
- The pipeline is robust to user errors: steps can be run out of order or multiple times without crashing.

## API Usage

### 1. Create a Pipeline

**Auto Mode (recommended):**
```python
from degirum_face import FaceRecognition
face_rec = FaceRecognition.auto("hailo8", inference_host_address="@local")
```

**From Config:**
```python
face_rec = FaceRecognition.from_config(
    detector_model="yolo_v8n_face_detection",
    embedder_model="face_embedding_mobilenet",
    inference_host_address="@cloud"
)
```

**Custom:**
```python
from degirum_face import PipelineModelConfig
config = PipelineModelConfig(
    detector_model="custom_detector",
    embedder_model="custom_embedder",
    zoo_url="https://custom.zoo.url",
    inference_host_address="192.168.1.100"
)
face_rec = FaceRecognition.custom(config)
```

### 2. Stepwise Pipeline Example


```python
det_res = face_rec.detect_faces("image.jpg")
detections = det_res.results
face_rec.filter_faces(detections)  # Optional, explicit filtering (annotation-based)
aligned = face_rec.align_faces(det_res.image, detections)
embeddings = face_rec.get_face_embeddings(aligned)
identities = face_rec.get_identities(detections, embeddings)
for det in identities:
    print(det["label"], det["similarity"])
```

**Note:**
- Filtering only marks detections as rejected; all detections are always passed through the pipeline.
- Each step is robust to missing or repeated processing and will not crash if run out of order or multiple times.

### 3. High-Level Identification

```python
results_obj = face_rec.identify_faces("image.jpg")
detections = results_obj.results
for det in detections:
    print(det["label"], det["similarity"])
# You can also access results_obj.image_overlay for visualization
```

### 4. Enrollment (Add a Person to the Database)

```python
face_rec.enroll("Alice", ["alice1.jpg", "alice2.jpg"])
```

### 5. Verification (1:1 Face Match)

```python
is_match, similarity = face_rec.verify("alice1.jpg", "alice2.jpg")
print("Match" if is_match else "No match", similarity)
```


## Method Reference (face_recognition.py)

- `FaceRecognition.auto(hardware, inference_host_address, ...)` – Auto-select models for your hardware.
- `FaceRecognition.from_config(detector_model, embedder_model, inference_host_address, ...)` – Use specific models from config.
- `FaceRecognition.custom(config, ...)` – Full control with a PipelineModelConfig.
- `detect_faces(image)` – Detect faces in an image. Returns results object with `.image` and `.results` (detections).
- `filter_faces(detections)` – Annotate detections with `face_rejected` and `reject_reason` (annotation-based, does not remove detections).
- `align_faces(image, detections)` – Align and crop faces using detection landmarks. Skips rejected faces.
- `get_face_embeddings(aligned_faces)` – Get normalized embeddings for aligned faces.
- `get_identities(detections, embeddings)` – Assign labels to detections using the face database.
- `identify_faces(image)` – Full pipeline: detect, filter, align, embed, and identify in one call.
- `enroll(person_name, image_list)` – Add a person to the database with multiple images.
- `verify(image1, image2)` – 1:1 face verification. Returns (is_match, similarity).

## Example: Run the Web App

```bash
git clone https://github.com/DeGirum/degirum_face
cd degirum_face/examples
python face_tracking_web_app.py
# Then open http://localhost:8080 in your browser
```

## Advanced Topics

- **Database:** Uses LanceDB for fast, deduplicated face search. Default path: `face_recognition.lance`.
- **Thresholds:** `similarity_threshold` controls match strictness (0-1, default 0.3 or 0.6).
- **Logging:** Enable or disable detailed logs with `enable_logging`.
- **Custom Models:** Use your own models by providing a custom config.

## License

Copyright DeGirum Corporation 2025

