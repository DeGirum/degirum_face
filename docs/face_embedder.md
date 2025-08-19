# Face Embedder Module

The FaceEmbedder module provides a clean, easy-to-use API for generating face embeddings and performing face verification using DeGirum AI models.

## Overview

Face embeddings are numerical vectors that represent unique characteristics of a face. These embeddings can be used for:

- **Face Recognition**: Identify specific individuals
- **Face Verification**: Verify if two images contain the same person
- **Face Clustering**: Group similar faces together
- **Face Search**: Find similar faces in a database

## Quick Start

### Auto Mode (Recommended)

```python
import degirum_face

# Create embedder with automatic model selection
embedder = degirum_face.FaceEmbedder("hailo8")

# Generate embedding for a single image
result = embedder.embed("person.jpg")
embedding_vector = embedder.extract_embedding_vector(result)

# Verify if two images contain the same person
verification = embedder.verify_faces(
    "person1_photo1.jpg", 
    "person1_photo2.jpg",
    threshold=0.7
)

print(f"Same person: {verification['is_same_person']}")
print(f"Similarity: {verification['similarity_score']:.3f}")
```

### Factory Methods

```python
# Explicit auto mode
embedder = degirum_face.FaceEmbedder.auto(hardware="hailo8")

# Custom model specification
embedder = degirum_face.FaceEmbedder.custom(
    model_name="arcface_mobilefacenet--112x112_quant_hailort_hailo8_1",
    zoo_url="degirum/hailo"
)

# Configuration-based
embedder = degirum_face.FaceEmbedder.from_config(
    hardware="hailo8",
    model_name="arcface_mobilefacenet--112x112_quant_hailort_hailo8_1"
)
```

### Convenience Functions

```python
# Single embedding (for simple use cases)
result = degirum_face.embed_face("image.jpg", hardware="hailo8")

# Quick verification
verification = degirum_face.verify_faces(
    "person1.jpg", "person2.jpg", 
    hardware="hailo8", threshold=0.7
)
```

## Core Methods

### Embedding Generation

```python
# Single image
result = embedder.embed("image.jpg")

# Batch processing
results = embedder.embed_batch(["img1.jpg", "img2.jpg", "img3.jpg"])

# Streaming (for video processing)
for result in embedder.embed_stream(video_frame_generator):
    embedding = embedder.extract_embedding_vector(result)
    # Process embedding...
```

### Embedding Comparison

```python
# Compare two embeddings
similarity = embedder.compare_embeddings(
    embedding1, embedding2, 
    metric="cosine"  # "cosine", "euclidean", or "dot_product"
)

# Face verification with threshold
verification = embedder.verify_faces(
    "image1.jpg", "image2.jpg",
    threshold=0.7,
    metric="cosine"
)
```

## Similarity Metrics

### Cosine Similarity (Recommended)
- Range: -1 to 1 (1 = identical, 0 = orthogonal, -1 = opposite)
- Best for normalized embeddings
- Most commonly used in face recognition

### Euclidean Distance
- Converted to similarity: 1.0 / (1.0 + distance)
- Range: 0 to 1 (higher = more similar)
- Good for unnormalized embeddings

### Dot Product
- Range depends on embedding magnitude
- Fast computation
- Suitable when embeddings are pre-normalized

## Supported Hardware

Check available hardware:

```python
hardware_list = degirum_face.FaceEmbedder.get_supported_hardware()
print(f"Supported hardware: {hardware_list}")

# Get available models for specific hardware
models = degirum_face.FaceEmbedder.get_available_models("hailo8")
print(f"Available models: {models}")
```

Current supported hardware includes:
- `hailo8`: Hailo-8 AI processor
- `hailo8l`: Hailo-8L AI processor  
- `degirum_orca`: DeGirum Orca AI processor
- `cpu`: CPU inference
- And more...

## Model Information

```python
# Get detailed model information
info = embedder.get_model_info()
print(f"Model: {info['model_name']}")
print(f"Hardware: {info['hardware']}")
print(f"Task: {info['task']}")
```

## Advanced Usage

### Building a Face Database

```python
import numpy as np

# Initialize embedder
embedder = degirum_face.FaceEmbedder("hailo8")

# Build database of known faces
known_faces = {}
for person_name in ["alice", "bob", "charlie"]:
    result = embedder.embed(f"{person_name}_reference.jpg")
    known_faces[person_name] = embedder.extract_embedding_vector(result)

# Identify unknown face
unknown_result = embedder.embed("unknown_person.jpg")
unknown_embedding = embedder.extract_embedding_vector(unknown_result)

# Find best match
best_match = None
best_similarity = 0.0

for name, known_embedding in known_faces.items():
    similarity = embedder.compare_embeddings(
        unknown_embedding, known_embedding
    )
    if similarity > best_similarity:
        best_similarity = similarity
        best_match = name

if best_similarity > 0.7:  # Threshold
    print(f"Identified: {best_match} (similarity: {best_similarity:.3f})")
else:
    print("Unknown person")
```

### Real-time Video Processing

```python
import cv2

embedder = degirum_face.FaceEmbedder("hailo8")

def process_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Generate embedding for current frame
        # (assumes frame contains a face)
        result = embedder.embed(frame)
        embedding = embedder.extract_embedding_vector(result)
        
        # Process embedding (compare with database, etc.)
        # ...
        
    cap.release()
```

## Error Handling

```python
try:
    embedder = degirum_face.FaceEmbedder("invalid_hardware")
except ValueError as e:
    print(f"Hardware error: {e}")

try:
    result = embedder.embed("nonexistent_image.jpg")
except Exception as e:
    print(f"Processing error: {e}")
```

## Performance Tips

1. **Reuse embedder instances**: Creating a new embedder for each operation is inefficient
2. **Use batch processing**: For multiple images, use `embed_batch()` instead of multiple `embed()` calls
3. **Choose appropriate hardware**: Hardware accelerators (Hailo8, Orca) are much faster than CPU
4. **Pre-normalize embeddings**: Store normalized embeddings for faster comparisons
5. **Adjust thresholds**: Tune similarity thresholds based on your specific use case

## Integration with Face Detection

```python
# Combine with face detection for complete pipeline
detector = degirum_face.FaceDetector("hailo8")
embedder = degirum_face.FaceEmbedder("hailo8")

def process_image_with_faces(image_path):
    # First detect faces
    detection_result = detector.detect(image_path)
    
    # Extract face regions and generate embeddings
    for face in detection_result.faces:  # Assuming faces are detected
        face_crop = extract_face_crop(image_path, face.bbox)
        embedding_result = embedder.embed(face_crop)
        embedding = embedder.extract_embedding_vector(embedding_result)
        
        # Process embedding...
```

## See Also

- [Face Detector Module](face_detector.py) - For face detection
- [Examples](../examples/face_embedder_examples.py) - Complete usage examples
- [Tests](../tests/test_face_embedder.py) - Unit tests and validation
