# Face Recognition Pipeline Documentation

## Overview

The DeGirum Face Recognition Pipeline provides a comprehensive, production-ready solution for face enrollment, verification, and identification. It integrates face detection, embedding generation, and database management into a unified API that handles quality control, performance optimization, and error management automatically.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Face          │    │   Face           │    │   Database      │
│   Detection     │───▶│   Embedding      │───▶│   Management    │
│                 │    │   Generation     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Quality       │    │   Face           │    │   Similarity    │
│   Assessment    │    │   Alignment      │    │   Search        │
│                 │    │   & Cropping     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Core Components

1. **FaceRecognition**: Main orchestrator class
2. **FaceDetector**: Face detection with auto model selection
3. **FaceEmbedder**: Face embedding generation
4. **ReID_Database**: Persistent embedding storage
5. **Face Utils**: Alignment, cropping, and quality assessment

## Key Features

### ✅ **Auto-Configuration**
- Automatic hardware detection and model selection
- Optimized defaults for different use cases
- Seamless integration with DeGirum model zoo

### ✅ **Quality Control**
- Automatic face quality assessment
- Configurable quality thresholds
- Multi-metric quality scoring (size, pose, sharpness, lighting)

### ✅ **Robust Database Management**
- Persistent embedding storage with LanceDB
- Automatic deduplication
- Efficient similarity search
- Database statistics and monitoring

### ✅ **Flexible Recognition Modes**
- **1:1 Verification**: Is this person X?
- **1:N Identification**: Who is this person?
- **Batch Processing**: Process multiple images efficiently

### ✅ **Production Ready**
- Comprehensive error handling
- Detailed logging and monitoring
- Performance optimization
- Thread-safe operations

## Installation

### Prerequisites

1. **DeGirum SDK**: Face detection and embedding models
2. **Hardware Support**: Hailo8, CUDA, or CPU
3. **Python Dependencies**: NumPy, OpenCV, LanceDB

### Basic Installation

```bash
# Install the degirum_face package
pip install -e .

# Verify installation
python -c "from degirum_face import FaceRecognition; print('✓ Installation successful')"
```

### Hardware Setup

#### Hailo8 (Recommended for Production)
```bash
# Install Hailo8 drivers and SDK
# Follow DeGirum Hailo8 setup guide
```

#### CUDA (For GPU acceleration)
```bash
# Install CUDA drivers and DeGirum CUDA support
# Follow DeGirum CUDA setup guide
```

#### CPU (For development/testing)
```bash
# CPU support is included by default
# No additional setup required
```

## Quick Start

### Basic Usage

```python
from degirum_face import FaceRecognition

# Initialize with auto-configuration
face_rec = FaceRecognition(
    hardware="hailo8",  # or "cuda", "cpu", None for auto
    db_path="my_faces.lance"
)

# Enroll a person
result = face_rec.enroll_person(
    person_id="john_doe",
    images=["john1.jpg", "john2.jpg", "john3.jpg"],
    attributes={"name": "John Doe", "department": "Engineering"}
)

# Verify identity (1:1)
is_match, confidence = face_rec.verify_person("test.jpg", "john_doe")

# Identify person (1:N)  
person_id, confidence = face_rec.identify_person("unknown.jpg")
```

### Command Line Interface

```bash
# Initialize database
python face_recognition_cli.py init --db faces.lance

# Enroll person
python face_recognition_cli.py enroll john_doe \
    --photos john1.jpg john2.jpg john3.jpg \
    --name "John Doe" --dept Engineering

# Verify identity
python face_recognition_cli.py verify john_doe --image test.jpg

# Identify person
python face_recognition_cli.py identify --image unknown.jpg --top-n 3

# List enrolled persons
python face_recognition_cli.py list

# Show statistics
python face_recognition_cli.py stats
```

## API Reference

### FaceRecognition Class

#### Constructor

```python
FaceRecognition(
    hardware: Optional[str] = None,
    *,
    db_path: str = "face_recognition.lance",
    similarity_threshold: float = 0.6,
    quality_threshold: float = 0.5,
    max_faces_per_image: int = 1,
    embedding_size: int = 112,
    zoo_url: str = "degirum/public",
    token: Optional[str] = None,
    face_detector_model: Optional[str] = None,
    face_embedder_model: Optional[str] = None,
    enable_logging: bool = True
)
```

**Parameters:**

- `hardware`: Target hardware ("hailo8", "cuda", "cpu", None for auto-detect)
- `db_path`: Path to face database file (LanceDB format)
- `similarity_threshold`: Minimum similarity for face matching (0-1)
- `quality_threshold`: Minimum quality score for enrollment (0-1)
- `max_faces_per_image`: Maximum faces to process per image
- `embedding_size`: Target size for face alignment (112 recommended)
- `zoo_url`: Model zoo URL for loading AI models
- `token`: Authentication token for model zoo access
- `face_detector_model`: Override detector model name
- `face_embedder_model`: Override embedder model name
- `enable_logging`: Enable detailed logging

#### Methods

##### enroll_person()

```python
enroll_person(
    person_id: str,
    images: Union[str, List[str], np.ndarray, List[np.ndarray]],
    *,
    attributes: Optional[Dict[str, Any]] = None,
    replace_existing: bool = False,
    min_faces_required: int = 1,
    max_faces_to_enroll: int = 10
) -> EnrollmentResult
```

Enroll a person with quality-controlled face embeddings.

**Parameters:**
- `person_id`: Unique identifier for the person
- `images`: Single image or list of images (file paths, URLs, or numpy arrays)
- `attributes`: Optional person attributes (name, department, etc.)
- `replace_existing`: If True, replace existing person data
- `min_faces_required`: Minimum faces needed for successful enrollment
- `max_faces_to_enroll`: Maximum faces to store per person

**Returns:** `EnrollmentResult` with enrollment statistics

##### verify_person()

```python
verify_person(
    image: Union[str, np.ndarray],
    person_id: str,
    *,
    confidence_threshold: Optional[float] = None
) -> Tuple[bool, float]
```

Verify if a face image matches a specific person (1:1 verification).

**Parameters:**
- `image`: Image containing face (file path, URL, or numpy array)
- `person_id`: ID of person to verify against
- `confidence_threshold`: Override default similarity threshold

**Returns:** Tuple of (is_match: bool, confidence: float)

##### identify_person()

```python
identify_person(
    image: Union[str, np.ndarray],
    *,
    confidence_threshold: Optional[float] = None,
    return_top_n: int = 1
) -> Union[Tuple[Optional[str], float], List[Tuple[str, float]]]
```

Identify a person from a face image (1:N identification).

**Parameters:**
- `image`: Image containing face (file path, URL, or numpy array)
- `confidence_threshold`: Override default similarity threshold
- `return_top_n`: Number of top candidates to return

**Returns:** 
- If `return_top_n=1`: Tuple of (person_id: Optional[str], confidence: float)
- If `return_top_n>1`: List of (person_id: str, confidence: float) tuples

##### get_enrolled_persons()

```python
get_enrolled_persons() -> Dict[str, Dict[str, Any]]
```

Get list of all enrolled persons and their attributes.

**Returns:** Dictionary mapping person_id to person attributes

##### get_database_stats()

```python
get_database_stats() -> Dict[str, Any]
```

Get statistics about the face database.

**Returns:** Dictionary with database statistics

### Result Classes

#### EnrollmentResult

```python
@dataclass
class EnrollmentResult:
    person_id: str
    num_faces_processed: int
    num_faces_enrolled: int  
    num_faces_rejected: int
    quality_scores: List[float]
    embedding_count: int
```

#### RecognitionResult

```python
@dataclass
class RecognitionResult:
    person_id: Optional[str]
    confidence: float
    embedding: np.ndarray
    quality: FaceQualityMetrics
    bbox: Optional[List[float]] = None
    landmarks: Optional[List[np.ndarray]] = None
```

#### FaceQualityMetrics

```python
@dataclass
class FaceQualityMetrics:
    face_size: float
    confidence: float
    landmark_quality: float
    frontal_score: float
    sharpness: float
    brightness: float
    overall_quality: float
```

## Configuration Guide

### Hardware Configuration

#### Hailo8 (Production Recommended)
```python
face_rec = FaceRecognition(
    hardware="hailo8",
    similarity_threshold=0.7,  # Higher threshold for security
    quality_threshold=0.6      # Higher quality for reliability
)
```

#### CUDA (GPU Acceleration)
```python
face_rec = FaceRecognition(
    hardware="cuda",
    max_faces_per_image=3,     # Process multiple faces efficiently
    enable_logging=False       # Reduce overhead for speed
)
```

#### CPU (Development/Testing)
```python
face_rec = FaceRecognition(
    hardware="cpu",
    similarity_threshold=0.6,  # Standard threshold
    quality_threshold=0.5      # Moderate quality requirements
)
```

### Threshold Configuration

#### Security Applications (High Precision)
```python
face_rec = FaceRecognition(
    similarity_threshold=0.8,  # Very strict matching
    quality_threshold=0.7,     # High quality faces only
    min_faces_required=3       # Multiple enrollment photos
)
```

#### Consumer Applications (Balanced)
```python
face_rec = FaceRecognition(
    similarity_threshold=0.6,  # Balanced accuracy/convenience
    quality_threshold=0.5,     # Moderate quality acceptance
    min_faces_required=1       # Single photo enrollment
)
```

#### Development/Testing (Permissive)
```python
face_rec = FaceRecognition(
    similarity_threshold=0.4,  # Lower threshold for testing
    quality_threshold=0.3,     # Accept lower quality faces
    min_faces_required=1       # Single photo enrollment
)
```

## Use Case Examples

### Employee Access Control

```python
class EmployeeAccessControl:
    def __init__(self):
        self.face_rec = FaceRecognition(
            hardware="hailo8",
            db_path="employees.lance",
            similarity_threshold=0.75,  # High security
            quality_threshold=0.6
        )
    
    def enroll_employee(self, employee_id, photos, employee_data):
        return self.face_rec.enroll_person(
            person_id=employee_id,
            images=photos,
            attributes=employee_data,
            min_faces_required=2,
            max_faces_to_enroll=5
        )
    
    def check_access(self, camera_image):
        person_id, confidence = self.face_rec.identify_person(camera_image)
        
        if person_id and confidence >= 0.75:
            # Grant access
            employees = self.face_rec.get_enrolled_persons()
            employee = employees.get(person_id)
            return {
                "access_granted": True,
                "employee": employee,
                "confidence": confidence
            }
        else:
            # Deny access
            return {
                "access_granted": False,
                "reason": "Unknown person or low confidence",
                "confidence": confidence
            }
```

### Photo Organization

```python
class PhotoOrganizer:
    def __init__(self, library_path):
        self.face_rec = FaceRecognition(
            db_path="photo_faces.lance",
            similarity_threshold=0.6,
            quality_threshold=0.4  # Accept casual photos
        )
        self.library_path = Path(library_path)
    
    def learn_faces_from_named_folders(self, training_path):
        """Learn faces from manually organized photos."""
        for person_folder in Path(training_path).iterdir():
            if person_folder.is_dir():
                person_name = person_folder.name
                person_id = person_name.lower().replace(" ", "_")
                
                photos = list(person_folder.glob("*.jpg"))
                if photos:
                    self.face_rec.enroll_person(
                        person_id=person_id,
                        images=[str(p) for p in photos],
                        attributes={"name": person_name}
                    )
    
    def organize_photos(self):
        """Automatically organize photos by person."""
        for photo_path in self.library_path.rglob("*.jpg"):
            try:
                person_id, confidence = self.face_rec.identify_person(str(photo_path))
                
                if person_id and confidence >= 0.6:
                    # Move to person folder
                    persons = self.face_rec.get_enrolled_persons()
                    person_name = persons[person_id].get("name", person_id)
                    
                    dest_folder = self.library_path / "organized" / person_name
                    dest_folder.mkdir(parents=True, exist_ok=True)
                    
                    dest_path = dest_folder / photo_path.name
                    photo_path.rename(dest_path)
                    
            except Exception as e:
                print(f"Error processing {photo_path}: {e}")
```

### Visitor Management

```python
import datetime

class VisitorManagement:
    def __init__(self):
        self.face_rec = FaceRecognition(
            db_path="visitors.lance",
            similarity_threshold=0.65
        )
    
    def register_visitor(self, visitor_name, host_employee, photos):
        visitor_id = f"VISITOR_{datetime.date.today()}_{visitor_name.replace(' ', '_')}"
        
        attributes = {
            "name": visitor_name,
            "type": "visitor",
            "host_employee": host_employee,
            "visit_date": str(datetime.date.today()),
            "check_in_time": str(datetime.datetime.now())
        }
        
        return self.face_rec.enroll_person(
            person_id=visitor_id,
            images=photos,
            attributes=attributes
        )
    
    def identify_visitor(self, camera_image):
        person_id, confidence = self.face_rec.identify_person(camera_image)
        
        if person_id:
            visitors = self.face_rec.get_enrolled_persons()
            visitor_info = visitors.get(person_id)
            
            if visitor_info and visitor_info.get("type") == "visitor":
                return {
                    "identified": True,
                    "visitor_info": visitor_info,
                    "confidence": confidence
                }
        
        return {"identified": False, "confidence": confidence}
```

## Performance Optimization

### Batch Processing

```python
def process_images_efficiently(face_rec, image_paths, batch_size=32):
    """Process multiple images in batches for better performance."""
    
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        
        batch_results = []
        for image_path in batch:
            try:
                person_id, confidence = face_rec.identify_person(image_path)
                batch_results.append((image_path, person_id, confidence))
            except Exception as e:
                batch_results.append((image_path, None, 0.0))
        
        results.extend(batch_results)
        print(f"Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")
    
    return results
```

### Hardware Optimization

```python
# High-throughput configuration for Hailo8
face_rec = FaceRecognition(
    hardware="hailo8",
    max_faces_per_image=3,  # Process multiple faces
    enable_logging=False,   # Reduce logging overhead
    quality_threshold=0.6   # Balance quality vs speed
)

# Multi-GPU configuration (when supported)
face_rec = FaceRecognition(
    hardware="cuda",
    face_detector_model="yolo_v8n_face_det",  # Faster detector
    face_embedder_model="mobilefacenet",      # Lighter embedder
)
```

### Database Optimization

```python
# Monitor database performance
stats = face_rec.get_database_stats()
print(f"Database has {stats['total_persons']} persons")
print(f"Average embeddings per person: {stats['avg_embeddings_per_person']:.1f}")

# Optimize enrollment for large databases
face_rec.enroll_person(
    person_id="new_person",
    images=photos,
    max_faces_to_enroll=3,  # Limit embeddings per person
    min_faces_required=1    # Reduce enrollment requirements
)
```

## Error Handling and Debugging

### Common Issues

#### No Faces Detected
```python
try:
    person_id, confidence = face_rec.identify_person("image.jpg")
except RuntimeError as e:
    if "No faces detected" in str(e):
        print("Solution: Ensure image contains visible faces")
        print("- Check image quality and lighting")
        print("- Try different image formats")
        print("- Verify face is not too small or obscured")
```

#### Low Quality Faces
```python
try:
    result = face_rec.enroll_person("person_id", ["low_quality.jpg"])
except ValueError as e:
    if "Insufficient quality faces" in str(e):
        print("Solution: Improve photo quality")
        print("- Use higher resolution images")
        print("- Ensure good lighting")
        print("- Face should be frontal and unobscured")
        print("- Try multiple photos from different angles")
```

#### Database Connection Issues
```python
try:
    face_rec = FaceRecognition(db_path="invalid/path/faces.lance")
except ConnectionError as e:
    print("Solution: Check database path and permissions")
    print("- Ensure directory exists and is writable")
    print("- Check file system permissions")
    print("- Verify disk space availability")
```

### Diagnostic Tools

```python
def diagnose_face_recognition(face_rec, test_image):
    """Comprehensive diagnostic for face recognition issues."""
    
    print("🔍 Diagnosing face recognition system...")
    
    # Test face detection
    try:
        detection_result = face_rec.detector.detect(test_image)
        if detection_result.results:
            print(f"✅ Face detection: {len(detection_result.results)} faces found")
            
            for i, face in enumerate(detection_result.results):
                bbox = face.bbox
                face_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                print(f"  Face {i+1}: size={face_size:.0f}px, confidence={getattr(face, 'confidence', 'N/A')}")
        else:
            print("❌ Face detection: No faces found")
            return
    except Exception as e:
        print(f"❌ Face detection failed: {e}")
        return
    
    # Test embedding generation
    try:
        person_id, confidence = face_rec.identify_person(test_image)
        print(f"✅ Embedding generation: Working")
        
        if person_id:
            print(f"✅ Person identified: {person_id} (confidence: {confidence:.3f})")
        else:
            print(f"ℹ️  Person not identified (confidence: {confidence:.3f})")
            
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
    
    # Test database
    try:
        stats = face_rec.get_database_stats()
        print(f"✅ Database: {stats['total_persons']} persons, {stats['total_embeddings']} embeddings")
    except Exception as e:
        print(f"❌ Database access failed: {e}")

# Usage
diagnose_face_recognition(face_rec, "test_image.jpg")
```

### Performance Monitoring

```python
import time
from contextlib import contextmanager

@contextmanager
def timing(operation):
    """Context manager for timing operations."""
    start = time.time()
    yield
    end = time.time()
    print(f"{operation}: {end - start:.3f}s")

# Monitor operation performance
with timing("Enrollment"):
    face_rec.enroll_person("test", ["photo1.jpg", "photo2.jpg"])

with timing("Verification"):
    is_match, confidence = face_rec.verify_person("test.jpg", "test")

with timing("Identification"):
    person_id, confidence = face_rec.identify_person("unknown.jpg")
```

## Security Considerations

### Production Deployment

1. **Secure Database Storage**
   - Use encrypted file systems for database files
   - Implement database access controls
   - Regular database backups

2. **Network Security**
   - Secure model zoo connections (HTTPS)
   - API token management
   - Network access controls

3. **Privacy Protection**
   - Implement data retention policies
   - Face embedding anonymization
   - Audit logging for access

### Threshold Configuration for Security

```python
# High-security configuration
face_rec = FaceRecognition(
    similarity_threshold=0.8,   # Very strict matching
    quality_threshold=0.7,      # High quality faces only
    max_faces_per_image=1,      # Single face processing
    min_faces_required=3        # Multiple enrollment photos required
)

# Additional security measures
def secure_verification(face_rec, image, person_id):
    """Enhanced verification with additional security checks."""
    
    # Multiple verification attempts
    attempts = 3
    matches = 0
    
    for _ in range(attempts):
        is_match, confidence = face_rec.verify_person(image, person_id)
        if is_match and confidence >= 0.8:
            matches += 1
    
    # Require majority consensus
    return matches >= (attempts // 2 + 1)
```

## Troubleshooting Guide

### Installation Issues

**Issue**: ModuleNotFoundError for degirum_face
```bash
# Solution: Install in development mode
pip install -e .

# Or add to Python path
export PYTHONPATH=$PYTHONPATH:/path/to/degirum_face
```

**Issue**: DeGirum SDK not found
```bash
# Solution: Install DeGirum SDK
pip install degirum

# Verify installation
python -c "import degirum; print(degirum.__version__)"
```

### Runtime Issues

**Issue**: "No faces detected in image"
- Check image quality and resolution
- Ensure faces are visible and not obscured
- Try different lighting conditions
- Verify image format is supported

**Issue**: "Insufficient quality faces for enrollment"
- Use higher resolution photos
- Ensure good lighting conditions
- Take photos from multiple angles
- Reduce quality_threshold parameter

**Issue**: "Database connection failed"
- Check file permissions on database directory
- Verify disk space availability
- Ensure LanceDB dependencies are installed

### Performance Issues

**Issue**: Slow recognition performance
- Use Hailo8 hardware for best performance
- Reduce max_faces_per_image for faster processing
- Disable detailed logging in production
- Consider batch processing for multiple images

**Issue**: High memory usage
- Limit max_faces_to_enroll per person
- Use CPU hardware for memory-constrained environments
- Process images in smaller batches

## Support and Documentation

### Additional Resources

- **DeGirum Documentation**: Official SDK documentation
- **Model Zoo**: Available face detection and embedding models
- **Hardware Support**: Hailo8, CUDA, and CPU setup guides
- **Performance Benchmarks**: Hardware-specific performance data

### Getting Help

1. **Check the troubleshooting guide** for common issues
2. **Run diagnostic tools** to identify specific problems
3. **Review logs** for detailed error information
4. **Contact support** with specific error messages and system configuration

### Contributing

The face recognition pipeline is designed to be extensible:

- **Custom Models**: Add support for new face detection/embedding models
- **Quality Metrics**: Implement additional face quality assessment algorithms
- **Database Backends**: Add support for alternative database systems
- **Hardware Targets**: Extend support for new hardware platforms

---

*This documentation covers the core functionality of the DeGirum Face Recognition Pipeline. For the latest updates and advanced features, please refer to the official documentation and release notes.*
