# Multi-Face Identification Guide

The DeGirum Face Recognition system provides two complementary approaches for identifying people in images:

## Single-Face Identification vs Multi-Face Identification

### `identify_person()` - Single Best Face
**Purpose**: Identifies the highest quality face in an image
**Best for**: Authentication, ID verification, portrait photos

```python
from degirum_face import FaceRecognition

# Local inference on Hailo8 device
face_rec = FaceRecognition(
    hardware="hailo8",           # Target AI hardware device
    inference_host_address="@localhost"  # Run inference locally
)

# Cloud inference with CPU fallback  
face_rec = FaceRecognition(
    hardware="cpu",              # Target AI hardware device
    inference_host_address="@cloud"      # Run inference on cloud
)

# Remote inference on edge device
face_rec = FaceRecognition(
    hardware="hailo8",           # Target AI hardware device
    inference_host_address="192.168.1.100"  # Run inference on remote host
)

# Returns the best quality face identification
person_id, confidence = face_rec.identify_person("portrait.jpg")
```

**How it works**:
1. Detects all faces in the image
2. Evaluates quality metrics for each face (size, sharpness, frontal angle, etc.)
3. Selects the face with the highest overall quality score
4. Returns identification for that single face only

**Use cases**:
- Employee badge authentication
- Phone/device unlock with face ID
- Driver's license verification
- ATM authentication
- Access control systems
- Single-person portrait analysis

### `identify_all_persons()` - All Faces
**Purpose**: Identifies every person detected in an image
**Best for**: Group photos, surveillance, event photography

```python
from degirum_face import FaceRecognition

# Initialize for group photo processing
face_rec = FaceRecognition(
    hardware="hailo8",           # Target AI hardware device  
    inference_host_address="@cloud",     # Run inference on cloud
    max_faces_per_image=10       # Process up to 10 faces
)

# Returns identification results for all detected faces
results = face_rec.identify_all_persons("group_photo.jpg")

for result in results:
    face_index = result["face_index"]
    person_id = result["person_id"] 
    confidence = result["confidence"]
    bbox = result["bbox"]
    quality = result["quality"]
```

**How it works**:
1. Detects all faces in the image
2. Processes each face individually for identification
3. Returns detailed results for every face found
4. Includes face location, quality metrics, and identification results

**Use cases**:
- Photo tagging and organization
- Security camera monitoring
- Event attendee tracking
- Family photo organization
- Classroom attendance systems
- Social media auto-tagging

## Method Comparison

| Feature | `identify_person()` | `identify_all_persons()` |
|---------|-------------------|------------------------|
| **Faces Processed** | Best quality face only | All detected faces |
| **Return Type** | `(person_id, confidence)` | `List[Dict]` with detailed results |
| **Performance** | Faster (single face) | Slower (multiple faces) |
| **Memory Usage** | Lower | Higher for group photos |
| **Quality Control** | Automatic best selection | Per-face quality assessment |
| **Location Info** | Not provided | Bounding box for each face |

## Result Structure

### `identify_person()` Returns
```python
person_id, confidence = face_rec.identify_person("image.jpg")

# person_id: str or None (if no match found)
# confidence: float (0.0 to 1.0)
```

### `identify_all_persons()` Returns
```python
results = face_rec.identify_all_persons("image.jpg")

# Each result dict contains:
{
    "face_index": 0,                    # Face number (0-based)
    "bbox": [x, y, width, height],      # Face location
    "person_id": "john_doe",            # Identified person (or None)
    "confidence": 0.85,                 # Match confidence (0.0-1.0)
    "quality": FaceQualityMetrics(...), # Quality assessment
    "candidates": [                     # Top candidate matches
        ("john_doe", 0.85),
        ("jane_smith", 0.72)
    ],
    # Optional fields for rejected faces:
    "rejection_reason": "Quality too low",
    "error": "Processing failed"
}
```

## Configuration Options

### Processing Limits
```python
face_rec = FaceRecognition(
    max_faces_per_image=5,        # Process up to 5 faces per image
    quality_threshold=0.5,        # Minimum quality for processing
    similarity_threshold=0.75     # Minimum similarity for match
)
```

### Quality Assessment
The system evaluates multiple quality metrics:
- **Face size**: Larger faces are generally higher quality
- **Sharpness**: Blurry faces are rejected
- **Frontal angle**: Profile views score lower
- **Lighting**: Under/over-exposed faces score lower
- **Detection confidence**: How confident the detector is

## Practical Examples

### Example 1: ID Verification (Single-Face)
```python
def verify_id_document(photo_path, expected_person_id):
    """Verify an ID document photo matches enrolled person."""
    
    person_id, confidence = face_rec.identify_person(photo_path)
    
    if person_id == expected_person_id:
        print(f"✓ ID verified: {confidence:.2f} confidence")
        return True
    else:
        print(f"✗ ID mismatch or unknown person")
        return False
```

### Example 2: Group Photo Tagging (Multi-Face)
```python
def tag_group_photo(photo_path):
    """Tag all people in a group photo."""
    
    results = face_rec.identify_all_persons(photo_path)
    
    tags = []
    for result in results:
        if result["person_id"]:
            person_info = face_rec.get_enrolled_persons()[result["person_id"]]
            tags.append({
                "name": person_info.get("name", result["person_id"]),
                "location": result["bbox"],
                "confidence": result["confidence"]
            })
        else:
            tags.append({
                "name": "Unknown Person",
                "location": result["bbox"],
                "confidence": result["confidence"]
            })
    
    return tags
```

### Example 3: Security Monitoring (Multi-Face)
```python
def monitor_entrance(camera_frame):
    """Monitor entrance for authorized personnel."""
    
    results = face_rec.identify_all_persons(camera_frame)
    
    alerts = []
    for result in results:
        if result["person_id"]:
            person_info = face_rec.get_enrolled_persons()[result["person_id"]]
            access_level = person_info.get("access_level", "none")
            
            if access_level not in ["employee", "contractor", "visitor"]:
                alerts.append({
                    "type": "unauthorized_person",
                    "person_id": result["person_id"],
                    "location": result["bbox"]
                })
        else:
            alerts.append({
                "type": "unknown_person", 
                "location": result["bbox"],
                "confidence": result["confidence"]
            })
    
    return alerts
```

## Performance Considerations

### Memory Usage
- `identify_person()`: Processes only the best face, minimal memory overhead
- `identify_all_persons()`: Memory usage scales with number of faces detected

### Processing Time
- `identify_person()`: ~50-200ms for typical images
- `identify_all_persons()`: ~100-500ms depending on face count

### Optimization Tips
1. **Limit face count**: Set `max_faces_per_image` appropriately for your use case
2. **Quality thresholds**: Higher quality thresholds reduce processing of poor faces
3. **Image preprocessing**: Resize large images before processing
4. **Batch processing**: Process multiple images in batches when possible

## Best Practices

### When to Use Single-Face Identification
- ✅ User authentication/login systems
- ✅ ID document verification
- ✅ Access control with individual photos
- ✅ Mobile apps with limited processing power
- ✅ Real-time applications requiring fast response

### When to Use Multi-Face Identification  
- ✅ Photo organization and tagging
- ✅ Security surveillance systems
- ✅ Event photography and attendee tracking
- ✅ Group photo analysis
- ✅ Classroom or meeting attendance
- ✅ Social media applications

### Error Handling
```python
try:
    # Single face identification
    person_id, confidence = face_rec.identify_person("photo.jpg")
    
    # Multi-face identification
    results = face_rec.identify_all_persons("group.jpg")
    
except FileNotFoundError:
    print("Image file not found")
except ValueError as e:
    print(f"Invalid parameters: {e}")
except Exception as e:
    print(f"Processing error: {e}")
```

## CLI Usage

The command-line interface supports both modes:

```bash
# Single face identification (default)
python face_recognition_cli.py identify --image photo.jpg

# Multi-face identification
python face_recognition_cli.py identify --image group.jpg --all-faces

# With custom threshold
python face_recognition_cli.py identify --image photo.jpg --threshold 0.8 --all-faces
```

## Integration Examples

See the complete examples in:
- `examples/multi_face_identification_example.py` - Comprehensive demonstration
- `examples/face_recognition_example.py` - Basic usage patterns
- `examples/face_recognition_cli.py` - Command-line interface

The multi-face identification capability significantly expands the system's applicability to group scenarios while maintaining the efficiency of single-face identification for authentication use cases.
