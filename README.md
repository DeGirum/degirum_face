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

The pipeline is explicit and modular. Each step is a method:

1. **Face Detection:** Locate faces in an image.
2. **Face Alignment:** Crop and align faces for embedding.
3. **Face Embedding:** Generate feature vectors for each face.
4. **Identification:** Match embeddings to database and assign labels.

You can run the full pipeline in one call, or step by step for maximum control.

## API Usage

### 1. Create a Pipeline

**Auto Mode (recommended):**
```python
from degirum_face import FaceRecognition
face_rec = FaceRecognition.auto("hailo8", inference_host_address="@localhost")
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
aligned = face_rec.align_faces(det_res.image, det_res.results)
embeddings = face_rec.get_face_embeddings(aligned)
identities = face_rec.get_identities(det_res.results, embeddings)
for det in identities:
    print(det["label"], det["similarity"])
```

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
- `align_faces(image, detections)` – Align and crop faces using detection landmarks.
- `get_face_embeddings(aligned_faces)` – Get normalized embeddings for aligned faces.
- `get_identities(detections, embeddings)` – Assign labels to detections using the face database.
- `identify_faces(image)` – Full pipeline: detect, align, embed, and identify in one call.
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

