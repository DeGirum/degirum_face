# Face Recognition Pipeline Usage Guide

This guide demonstrates practical usage scenarios for the DeGirum Face Recognition Pipeline, covering enrollment, verification, identification, and database management.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Employee Access Control](#employee-access-control)
3. [Visitor Management](#visitor-management)
4. [Security Monitoring](#security-monitoring)
5. [Photo Organization](#photo-organization)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Setup

```python
from degirum_face import FaceRecognition

# Initialize with auto-configuration
face_rec = FaceRecognition(
    hardware="hailo8",  # or "cuda", "cpu", None for auto-detect
    db_path="my_faces.lance",
    similarity_threshold=0.6,
    quality_threshold=0.5
)
```

### Simple Enrollment and Recognition

```python
# Enroll a person with multiple photos
result = face_rec.enroll_person(
    person_id="john_doe",
    images=["john1.jpg", "john2.jpg", "john3.jpg"],
    attributes={"name": "John Doe", "department": "Engineering"}
)

print(f"Enrolled {result.num_faces_enrolled} faces for John")

# Verify identity (1:1)
is_match, confidence = face_rec.verify_person("test_image.jpg", "john_doe")
if is_match:
    print(f"Verified John with {confidence:.2f} confidence")

# Identify unknown person (1:N)
person_id, confidence = face_rec.identify_person("unknown_face.jpg")
if person_id:
    print(f"Identified as {person_id} with {confidence:.2f} confidence")
else:
    print("Unknown person detected")
```

## Employee Access Control

### Scenario: Building Entry System

```python
import cv2
from degirum_face import FaceRecognition

class EmployeeAccessControl:
    """Employee access control system using face recognition."""
    
    def __init__(self, db_path="employees.lance"):
        self.face_rec = FaceRecognition(
            hardware="hailo8",
            db_path=db_path,
            similarity_threshold=0.7,  # Higher threshold for security
            quality_threshold=0.6
        )
        
    def enroll_employee(self, employee_id, photos, employee_data):
        """Enroll a new employee with HR data."""
        
        attributes = {
            "name": employee_data["name"],
            "department": employee_data["department"],
            "employee_id": employee_id,
            "access_level": employee_data.get("access_level", "standard"),
            "hire_date": employee_data.get("hire_date"),
            "status": "active"
        }
        
        try:
            result = face_rec.enroll_person(
                person_id=employee_id,
                images=photos,
                attributes=attributes,
                min_faces_required=2,  # Require at least 2 good photos
                max_faces_to_enroll=5
            )
            
            print(f"✓ Enrolled {attributes['name']} successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to enroll {attributes['name']}: {e}")
            return False
    
    def check_access(self, camera_image, area_access_level="standard"):
        """Check if person in image has access to area."""
        
        try:
            # Identify person
            person_id, confidence = self.face_rec.identify_person(camera_image)
            
            if not person_id:
                return {
                    "access_granted": False,
                    "reason": "Unknown person",
                    "confidence": confidence
                }
            
            # Get employee details
            employees = self.face_rec.get_enrolled_persons()
            employee = employees.get(person_id)
            
            if not employee:
                return {
                    "access_granted": False,
                    "reason": "Employee not found in database",
                    "confidence": confidence
                }
            
            # Check employee status and access level
            if employee.get("status") != "active":
                return {
                    "access_granted": False,
                    "reason": f"Employee status: {employee.get('status')}",
                    "employee": employee,
                    "confidence": confidence
                }
            
            # Check access level (simplified)
            employee_access = employee.get("access_level", "standard")
            access_hierarchy = {"standard": 1, "elevated": 2, "admin": 3}
            
            required_level = access_hierarchy.get(area_access_level, 1)
            employee_level = access_hierarchy.get(employee_access, 1)
            
            if employee_level >= required_level:
                return {
                    "access_granted": True,
                    "employee": employee,
                    "confidence": confidence
                }
            else:
                return {
                    "access_granted": False,
                    "reason": f"Insufficient access level: {employee_access}",
                    "employee": employee,
                    "confidence": confidence
                }
                
        except Exception as e:
            return {
                "access_granted": False,
                "reason": f"System error: {e}",
                "confidence": 0.0
            }

# Usage example
access_control = EmployeeAccessControl()

# Enroll employees from HR database
employees_to_enroll = [
    {
        "id": "EMP001",
        "data": {
            "name": "Alice Johnson",
            "department": "Engineering",
            "access_level": "elevated",
            "hire_date": "2023-01-15"
        },
        "photos": ["alice_id.jpg", "alice_casual.jpg", "alice_profile.jpg"]
    },
    {
        "id": "EMP002", 
        "data": {
            "name": "Bob Smith",
            "department": "Marketing",
            "access_level": "standard",
            "hire_date": "2023-03-01"
        },
        "photos": ["bob_headshot.jpg", "bob_meeting.jpg"]
    }
]

for emp in employees_to_enroll:
    access_control.enroll_employee(emp["id"], emp["photos"], emp["data"])

# Check access for camera capture
camera_image = "door_camera_capture.jpg"
result = access_control.check_access(camera_image, area_access_level="elevated")

if result["access_granted"]:
    print(f"✓ Access granted to {result['employee']['name']}")
else:
    print(f"✗ Access denied: {result['reason']}")
```

## Visitor Management

### Scenario: Office Visitor Check-in

```python
import datetime
from degirum_face import FaceRecognition

class VisitorManagement:
    """Visitor management system with temporary enrollment."""
    
    def __init__(self):
        self.face_rec = FaceRecognition(
            db_path="visitors.lance",
            similarity_threshold=0.65
        )
        
    def register_visitor(self, visitor_name, host_employee, visit_purpose, photos):
        """Register a new visitor for the day."""
        
        visitor_id = f"VISITOR_{datetime.date.today().strftime('%Y%m%d')}_{visitor_name.replace(' ', '_')}"
        
        attributes = {
            "name": visitor_name,
            "type": "visitor",
            "host_employee": host_employee,
            "visit_purpose": visit_purpose,
            "visit_date": str(datetime.date.today()),
            "check_in_time": str(datetime.datetime.now()),
            "status": "checked_in"
        }
        
        try:
            result = self.face_rec.enroll_person(
                person_id=visitor_id,
                images=photos,
                attributes=attributes,
                min_faces_required=1
            )
            
            return {
                "success": True,
                "visitor_id": visitor_id,
                "badge_info": {
                    "name": visitor_name,
                    "host": host_employee,
                    "date": attributes["visit_date"],
                    "purpose": visit_purpose
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def identify_visitor(self, camera_image):
        """Identify visitor from camera image."""
        
        person_id, confidence = self.face_rec.identify_person(camera_image)
        
        if person_id:
            visitors = self.face_rec.get_enrolled_persons()
            visitor_info = visitors.get(person_id)
            
            if visitor_info and visitor_info.get("type") == "visitor":
                return {
                    "identified": True,
                    "visitor_id": person_id,
                    "visitor_info": visitor_info,
                    "confidence": confidence
                }
        
        return {"identified": False, "confidence": confidence}
    
    def checkout_visitor(self, visitor_id):
        """Mark visitor as checked out (in practice, update database)."""
        
        # In a real implementation, you would update the database
        # For now, just return success
        return {"success": True, "checkout_time": str(datetime.datetime.now())}

# Usage example
visitor_mgmt = VisitorManagement()

# Register visitor at reception
visitor_registration = visitor_mgmt.register_visitor(
    visitor_name="Dr. Sarah Wilson",
    host_employee="Alice Johnson (EMP001)",
    visit_purpose="Technical consultation",
    photos=["visitor_photo_front.jpg", "visitor_photo_side.jpg"]
)

if visitor_registration["success"]:
    print(f"Visitor registered: {visitor_registration['visitor_id']}")
    
    # Later, identify visitor at security checkpoint
    camera_capture = "security_camera.jpg"
    identification = visitor_mgmt.identify_visitor(camera_capture)
    
    if identification["identified"]:
        visitor_info = identification["visitor_info"]
        print(f"Visitor identified: {visitor_info['name']}")
        print(f"Host: {visitor_info['host_employee']}")
        print(f"Purpose: {visitor_info['visit_purpose']}")
    else:
        print("Unknown person at security checkpoint")
```

## Security Monitoring

### Scenario: Real-time Security Alerts

```python
import cv2
import threading
import time
from degirum_face import FaceRecognition

class SecurityMonitoring:
    """Real-time security monitoring with face recognition."""
    
    def __init__(self, authorized_db_path="authorized.lance"):
        self.face_rec = FaceRecognition(
            db_path=authorized_db_path,
            similarity_threshold=0.7
        )
        
        self.monitoring_active = False
        self.alert_callbacks = []
        
    def add_alert_callback(self, callback):
        """Add callback function for security alerts."""
        self.alert_callbacks.append(callback)
        
    def monitor_camera_feed(self, camera_source=0, check_interval=2.0):
        """Monitor camera feed for unauthorized persons."""
        
        cap = cv2.VideoCapture(camera_source)
        self.monitoring_active = True
        
        last_check_time = time.time()
        
        while self.monitoring_active:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = time.time()
            
            # Check faces at specified intervals
            if current_time - last_check_time >= check_interval:
                self._check_frame_for_threats(frame, current_time)
                last_check_time = current_time
                
            # Display frame with annotations
            cv2.imshow("Security Monitor", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def _check_frame_for_threats(self, frame, timestamp):
        """Check frame for unauthorized persons."""
        
        try:
            person_id, confidence = self.face_rec.identify_person(frame)
            
            if person_id:
                # Known person detected
                authorized_persons = self.face_rec.get_enrolled_persons()
                person_info = authorized_persons.get(person_id)
                
                alert_data = {
                    "timestamp": timestamp,
                    "person_id": person_id,
                    "person_info": person_info,
                    "confidence": confidence,
                    "frame": frame.copy()
                }
                
                if person_info.get("security_status") == "restricted":
                    # Restricted person detected
                    alert_data["alert_type"] = "RESTRICTED_PERSON"
                    alert_data["severity"] = "HIGH"
                    self._trigger_alert(alert_data)
                    
            else:
                # Unknown person detected
                if confidence > 0.3:  # Only alert if there's a detectable face
                    alert_data = {
                        "timestamp": timestamp,
                        "alert_type": "UNKNOWN_PERSON",
                        "severity": "MEDIUM",
                        "confidence": confidence,
                        "frame": frame.copy()
                    }
                    self._trigger_alert(alert_data)
                    
        except Exception as e:
            print(f"Error during security check: {e}")
            
    def _trigger_alert(self, alert_data):
        """Trigger security alert."""
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                print(f"Alert callback error: {e}")
                
    def stop_monitoring(self):
        """Stop security monitoring."""
        self.monitoring_active = False

# Alert handlers
def log_security_alert(alert_data):
    """Log security alert to file."""
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert_data["timestamp"]))
    alert_type = alert_data["alert_type"]
    severity = alert_data["severity"]
    
    log_message = f"[{timestamp}] {severity} ALERT: {alert_type}"
    
    if "person_id" in alert_data:
        person_info = alert_data["person_info"]
        log_message += f" - {person_info.get('name', 'Unknown')} ({alert_data['person_id']})"
    
    print(log_message)
    
    # In practice, write to log file
    with open("security_alerts.log", "a") as f:
        f.write(log_message + "\n")

def save_alert_image(alert_data):
    """Save frame image for security alert."""
    
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(alert_data["timestamp"]))
    alert_type = alert_data["alert_type"]
    
    filename = f"alert_{alert_type}_{timestamp}.jpg"
    cv2.imwrite(filename, alert_data["frame"])
    print(f"Alert image saved: {filename}")

# Usage example
security_monitor = SecurityMonitoring()

# Add alert handlers
security_monitor.add_alert_callback(log_security_alert)
security_monitor.add_alert_callback(save_alert_image)

# Enroll authorized personnel
authorized_staff = [
    {
        "id": "SEC001",
        "name": "Security Guard John",
        "security_status": "authorized",
        "photos": ["john_guard.jpg"]
    },
    {
        "id": "MAINT001", 
        "name": "Maintenance Worker Bob",
        "security_status": "restricted",  # Restricted access
        "photos": ["bob_maintenance.jpg"]
    }
]

for staff in authorized_staff:
    security_monitor.face_rec.enroll_person(
        staff["id"],
        staff["photos"],
        attributes={
            "name": staff["name"],
            "security_status": staff["security_status"],
            "type": "staff"
        }
    )

# Start monitoring (uncomment to run)
# security_monitor.monitor_camera_feed(camera_source=0)
```

## Photo Organization

### Scenario: Personal Photo Library Organization

```python
import os
from pathlib import Path
from collections import defaultdict
from degirum_face import FaceRecognition

class PhotoOrganizer:
    """Organize photo library by automatically identifying people."""
    
    def __init__(self, library_path, organized_path):
        self.library_path = Path(library_path)
        self.organized_path = Path(organized_path)
        self.organized_path.mkdir(exist_ok=True)
        
        self.face_rec = FaceRecognition(
            db_path=str(self.organized_path / "photo_faces.lance"),
            similarity_threshold=0.65,
            quality_threshold=0.4  # Lower threshold for casual photos
        )
        
    def learn_faces_from_named_folders(self, named_photos_path):
        """Learn faces from manually organized photos."""
        
        named_path = Path(named_photos_path)
        
        for person_folder in named_path.iterdir():
            if person_folder.is_dir():
                person_name = person_folder.name
                person_id = person_name.lower().replace(" ", "_")
                
                # Collect all photos for this person
                photo_files = []
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    photo_files.extend(person_folder.glob(ext))
                
                if photo_files:
                    print(f"Learning faces for {person_name}...")
                    
                    try:
                        result = self.face_rec.enroll_person(
                            person_id=person_id,
                            images=[str(p) for p in photo_files[:10]],  # Use up to 10 photos
                            attributes={"name": person_name, "source": "manual"},
                            min_faces_required=1
                        )
                        
                        print(f"  ✓ Learned {result.num_faces_enrolled} faces")
                        
                    except Exception as e:
                        print(f"  ✗ Failed to learn {person_name}: {e}")
    
    def organize_photo_library(self):
        """Automatically organize photos by identifying faces."""
        
        # Create output folders
        identified_path = self.organized_path / "identified"
        unknown_path = self.organized_path / "unknown"
        no_faces_path = self.organized_path / "no_faces"
        
        for path in [identified_path, unknown_path, no_faces_path]:
            path.mkdir(exist_ok=True)
        
        # Process all photos in library
        photo_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            photo_files.extend(self.library_path.rglob(ext))
        
        print(f"Processing {len(photo_files)} photos...")
        
        stats = defaultdict(int)
        
        for i, photo_path in enumerate(photo_files):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(photo_files)} photos...")
                
            try:
                # Identify person in photo
                person_id, confidence = self.face_rec.identify_person(str(photo_path))
                
                if person_id:
                    # Get person info
                    persons = self.face_rec.get_enrolled_persons()
                    person_info = persons.get(person_id)
                    person_name = person_info.get("name", person_id)
                    
                    # Create person folder
                    person_folder = identified_path / person_name
                    person_folder.mkdir(exist_ok=True)
                    
                    # Copy photo to person folder
                    dest_path = person_folder / photo_path.name
                    self._copy_photo(photo_path, dest_path)
                    
                    stats["identified"] += 1
                    
                elif confidence > 0.1:  # Face detected but not recognized
                    dest_path = unknown_path / photo_path.name
                    self._copy_photo(photo_path, dest_path)
                    stats["unknown_faces"] += 1
                    
                else:  # No face detected
                    dest_path = no_faces_path / photo_path.name
                    self._copy_photo(photo_path, dest_path)
                    stats["no_faces"] += 1
                    
            except Exception as e:
                print(f"    Error processing {photo_path.name}: {e}")
                stats["errors"] += 1
        
        # Print statistics
        print("\nOrganization complete!")
        print(f"  Identified persons: {stats['identified']}")
        print(f"  Unknown faces: {stats['unknown_faces']}")
        print(f"  No faces: {stats['no_faces']}")
        print(f"  Errors: {stats['errors']}")
        
    def _copy_photo(self, src_path, dest_path):
        """Copy photo file safely."""
        
        import shutil
        
        # Handle duplicate names
        if dest_path.exists():
            stem = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            
            while dest_path.exists():
                dest_path = dest_path.parent / f"{stem}_{counter}{suffix}"
                counter += 1
        
        shutil.copy2(src_path, dest_path)

# Usage example
organizer = PhotoOrganizer(
    library_path="~/Pictures/Unsorted",
    organized_path="~/Pictures/Organized"
)

# First, learn from manually organized sample photos
organizer.learn_faces_from_named_folders("~/Pictures/Training_Photos")

# Then organize the entire library
organizer.organize_photo_library()
```

## Performance Optimization

### Hardware Configuration

```python
# For high-throughput scenarios
face_rec = FaceRecognition(
    hardware="hailo8",  # Use Hailo8 for best performance
    similarity_threshold=0.65,
    quality_threshold=0.6,
    max_faces_per_image=3,  # Process up to 3 faces per image
    enable_logging=False    # Disable detailed logging for speed
)

# For batch processing
def process_batch_efficiently(face_rec, image_batch):
    """Process multiple images efficiently."""
    
    results = []
    
    # Process in batches to optimize GPU utilization
    batch_size = 32
    
    for i in range(0, len(image_batch), batch_size):
        batch = image_batch[i:i+batch_size]
        
        batch_results = []
        for image in batch:
            try:
                person_id, confidence = face_rec.identify_person(image)
                batch_results.append((image, person_id, confidence))
            except Exception as e:
                batch_results.append((image, None, 0.0))
        
        results.extend(batch_results)
        
        # Optional: print progress
        print(f"Processed {min(i+batch_size, len(image_batch))}/{len(image_batch)} images")
    
    return results
```

### Database Optimization

```python
# For large databases, consider periodic cleanup
def optimize_database(face_rec):
    """Optimize database for better performance."""
    
    # Get database statistics
    stats = face_rec.get_database_stats()
    print(f"Database has {stats['total_persons']} persons with {stats['total_embeddings']} embeddings")
    
    # Identify persons with too many embeddings
    persons_with_embeddings = stats.get('persons_with_embeddings', {})
    
    for person_id, (count, attrs) in persons_with_embeddings.items():
        if count > 10:  # Too many embeddings
            print(f"Person {person_id} has {count} embeddings (consider reducing)")
```

## Troubleshooting

### Common Issues and Solutions

```python
def diagnose_face_recognition_issues(face_rec, test_image):
    """Diagnose common face recognition issues."""
    
    print("Diagnosing face recognition issues...")
    
    try:
        # Test face detection
        detection_result = face_rec.detector.detect(test_image)
        
        if not detection_result.results:
            print("❌ Issue: No faces detected")
            print("Solutions:")
            print("  - Ensure image contains visible faces")
            print("  - Check image quality and lighting")
            print("  - Try different image formats")
            return
        
        print(f"✅ Face detection working: {len(detection_result.results)} faces found")
        
        # Test face quality
        for i, face_det in enumerate(detection_result.results):
            bbox = face_det.bbox
            face_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            print(f"  Face {i+1}:")
            print(f"    Size: {face_size:.0f} pixels")
            print(f"    Confidence: {getattr(face_det, 'confidence', 'N/A')}")
            
            if face_size < 5000:  # Small face
                print(f"    ⚠️  Face may be too small (minimum ~5000 pixels recommended)")
            
            if hasattr(face_det, 'landmarks') and face_det.landmarks:
                print(f"    ✅ Landmarks detected: {len(face_det.landmarks)}")
            else:
                print(f"    ❌ No landmarks detected (needed for alignment)")
        
        # Test embedding generation
        try:
            person_id, confidence = face_rec.identify_person(test_image)
            print(f"✅ Embedding generation working")
            
            if person_id:
                print(f"✅ Person identified: {person_id} (confidence: {confidence:.3f})")
            else:
                print(f"ℹ️  No person identified (confidence: {confidence:.3f})")
                
                if confidence < 0.1:
                    print("  Possible issues:")
                    print("    - Face quality too low")
                    print("    - Person not enrolled in database")
                    print("    - Similarity threshold too high")
                    
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            
    except Exception as e:
        print(f"❌ Face detection failed: {e}")
        print("Possible issues:")
        print("  - Model loading problems")
        print("  - Hardware compatibility")
        print("  - Image format not supported")

# Example usage for troubleshooting
diagnose_face_recognition_issues(face_rec, "problematic_image.jpg")
```

### Performance Monitoring

```python
import time
from contextlib import contextmanager

@contextmanager
def performance_timer(operation_name):
    """Context manager for timing operations."""
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{operation_name}: {end_time - start_time:.3f} seconds")

# Monitor performance of different operations
with performance_timer("Enrollment"):
    face_rec.enroll_person("test_person", ["photo1.jpg", "photo2.jpg"])

with performance_timer("Verification"):
    is_match, confidence = face_rec.verify_person("test.jpg", "test_person")

with performance_timer("Identification"):
    person_id, confidence = face_rec.identify_person("unknown.jpg")
```

This guide provides comprehensive examples for different real-world scenarios using the face recognition pipeline. Each scenario includes practical code that can be adapted for specific use cases.
