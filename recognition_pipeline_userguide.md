
# Face Recognition Pipeline: Modular Guide and Best Practices

## Overview

The face recognition pipeline is a modular, extensible system for detecting, tracking, and identifying faces in video streams or images. It is designed to be robust, efficient, and adaptable to a wide range of applications—from simple face detection to advanced multi-stage recognition with alerting and recording. Each stage of the pipeline (detection, alignment, embedding, filtering, tracking, recognition, and more) can be enabled, configured, or extended independently, allowing you to tailor the system to your specific needs. This guide explains the purpose and logic of each stage, provides best practices, and shows how to customize the pipeline for your use case.

## Pipeline Structure: Basic, Advanced, and Optional Stages

The face recognition pipeline is modular and can be adapted to different application needs. Stages can be grouped as follows:

- **Basic Pipeline:**
  - Face Detection
  - Face Alignment & Extraction
  - Face Embedding (Feature Extraction)
  - Face Recognition (Identification)
  - Annotation (labeling and output)

- **Advanced Pipeline:**
  - Face Filters (quality, region, or pose-based filtering)

- **Even More Advanced Pipeline:**
  - Face Tracking (assigns consistent IDs across frames)
  - Identity Confirmation (ensures stable identity assignment)
  - reID Scheduling (optimizes embedding computation for performance)

- **Optional Stages (Application-Dependent):**
  - Alerting (trigger notifications or actions based on recognition events)
  - Recording and Storing (save video, images, or metadata when alerts occur)

You can start with the basic pipeline and add advanced or optional stages as needed for your use case.

## Typical Pipeline Flow

1. **Input:** Video frame or image.
2. **Detection:** Find all faces in the frame.
3. **Tracking:** Assign or update a track ID for each face.
4. **Filtering:** Remove low-quality or irrelevant faces/tracks.
5. **reID Scheduling:** Determine which faces/tracks need new embeddings this frame.
6. **Alignment & Extraction:** Align faces and extract crops for those needing embedding.
7. **Embedding:** Extract feature vectors for each face needing embedding; reuse cached embeddings for others.
8. **Recognition:** Match faces to known identities.
9. **Identity Confirmation:** Confirm identities over multiple frames.
10. **Annotation:** Attach labels and output results.
11. **Alerting:** Trigger notifications or actions based on recognition/confirmation events.
12. **Recording and Storing:** Save video, images, or metadata when alerts are triggered.

## Pipeline Stages

### 1. Face Detection
- **Purpose:** Locate faces in each frame.
- **How it works:** Uses a deep learning model to output bounding boxes for all detected faces.
- **Output:** List of bounding boxes (coordinates) for each detected face.

### 2. Face Tracking
- **Purpose:** Assign a consistent ID to each face as it moves across frames.
- **How it works:** Associates new detections with existing tracks using spatial and appearance cues.
- **Output:** Each face is assigned an ID that persists across frames.

### 3. Face Filtering
- **Purpose:** Discard faces that are too small, poorly aligned, not frontal, or outside a region of interest.
- **How it works:** Applies a sequence of filters (see Face Filters System) to each detection/track.
- **Output:** List of faces/tracks that pass all filters.


### 4. reID Scheduling
- **Purpose:** Decide which faces/tracks need a new embedding computation in this frame.
- **How it works:** Applies reID scheduling logic to filtered tracks, selecting only those that require a new embedding (e.g., after a certain number of frames or if the face's status changes). This acts as an additional filter, optimizing performance by avoiding redundant embedding computations for stable faces.
- **Output:** List of tracks/faces that need embedding; others can reuse cached embeddings.

### 5. Face Alignment & Extraction
- **Purpose:** Normalize face orientation and scale for better recognition accuracy, and extract face crops for embedding.
- **How it works:** Uses facial landmarks (eyes, nose, mouth) to align faces to a canonical pose and extract the aligned face region, but only for those faces scheduled for embedding.
- **Output:** Aligned face crops and/or landmark coordinates for faces needing embedding.

### 6. Face Embedding (Feature Extraction)
- **Purpose:** Convert each face into a compact feature vector (embedding) for comparison.
- **How it works:** Runs a face embedding model on each aligned face crop (for those selected by reID expiration); other faces reuse cached embeddings.
- **Output:** Embedding vector for each face (recomputed only as needed).

### 7. Face Recognition (Identification)
- **Purpose:** Match each face embedding to a database of known identities.
- **How it works:** Compares embeddings to those in a reference database using a similarity metric (e.g., cosine similarity).
- **Output:** For each face, the best-matching identity (or 'unknown') and a confidence score.

### 8. Identity Confirmation
- **Purpose:** Ensure that a face is consistently recognized as the same identity before confirming it.
- **How it works:** Requires a face to be matched to the same identity for a configurable number of frames before marking it as confirmed (see Identity Confirmation Logic).
- **Output:** Faces are marked as confirmed or still in the process of identity confirmation.

### 9. Annotation
- **Purpose:** Attach human-readable labels and attributes to each face for downstream use (UI, alerts, logging).
- **How it works:** Assigns labels based on tracking and confirmation status.
- **Output:** Metadata for each face, including label, identity, and status.

### 10. Alerting (Optional)
- **Purpose:** Trigger notifications or actions when certain face recognition or identity confirmation events occur (e.g., known person detected, unknown person appears).
- **How it works:** Monitors annotated results and identity confirmation status to send alerts, log events, or trigger downstream actions based on configurable rules.
- **Output:** Alerts, notifications, or actions sent to external systems or users.

### 11. Recording and Storing (Optional)
- **Purpose:** Save video clips, images, or metadata when specific alert conditions are met (e.g., recognized person, unknown face, or other events).
- **How it works:** Triggered by the alerting stage, this component records relevant frames, video segments, or data and stores them for later review, audit, or evidence.
- **Output:** Saved media files or logs associated with alert events.

## Example Code Snippet

```python
# Simplified pseudocode for the core pipeline
for frame in video_stream:
    faces = detector.detect(frame)
    tracks = tracker.update(faces)
    filtered_tracks = [t for t in tracks if all(filter(t) for filter in filters)]
    tracks_to_embed = [t for t in filtered_tracks if reid_scheduling.should_embed(t)]
    aligned_faces = [align_and_extract(t) for t in tracks_to_embed]
    embeddings = [embedder(f) for f in aligned_faces]
    # Recognize identities for tracks needing embedding; reuse previous identities for others
    identities = recognizer.assign_identities(filtered_tracks, embeddings, tracks_to_embed)
    confirmed = confirmation_logic(filtered_tracks, identities)
    # Annotate gizmo operates on all detected faces
    annotated = annotate_gizmo(faces, identities, confirmed)
    # Optional: alerting based on annotated results
    alert_events = alerting(annotated)
    # Optional: recording and storing triggered by alerts
    recording_and_storing(alert_events)
    output(annotated)
```

## Understanding Face Detection

### What is Face Detection?

Face detection is the process of automatically locating human faces in images or video frames. It is the first step in the face recognition pipeline, providing the bounding boxes that define where faces are present in each frame.

### How Face Detection Works

- A deep learning model (e.g., SSD, YOLO, RetinaFace, or MTCNN) scans each frame to find regions that likely contain faces.
- The model outputs a list of bounding boxes, each with coordinates (x, y, width, height) and often a confidence score.
- Some detectors also provide facial landmarks (e.g., eyes, nose, mouth) for each detected face.

#### 1. Model Selection
- Choose a face detector based on your accuracy, speed, and hardware requirements.
- Some models are optimized for real-time performance, while others prioritize accuracy in challenging conditions.

#### 2. Detection Parameters
- **Confidence Threshold:** Minimum score for a detection to be considered valid.
- **Input Size:** The resolution at which the model processes frames can affect speed and accuracy.

#### 3. Output
- For each frame, the detector returns a list of bounding boxes and (optionally) landmarks.
- These detections are passed to the next stage (tracking).

### Why Use Face Detection?

- **Efficiency:** Focuses downstream processing only on regions containing faces.
- **Scalability:** Enables handling of multiple faces in a single frame.
- **Robustness:** Good detectors handle variations in pose, lighting, and occlusion.

### Example


Suppose a frame contains three people. The detector outputs three bounding boxes in (x1, y1, x2, y2) format:

- Box 1: (x1=100, y1=50, x2=180, y2=130)
- Box 2: (x1=300, y1=60, x2=375, y2=135)
- Box 3: (x1=500, y1=55, x2=590, y2=145)

These are then passed to the tracking stage.

> **Tip:** Adjust the confidence threshold to balance between missing faces (false negatives) and detecting non-faces (false positives). Lower thresholds may increase false positives; higher thresholds may miss small or low-quality faces.

## Understanding Face Tracking

### What is Face Tracking?

Face tracking is the process of assigning a consistent ID to each detected face as it moves across frames in a video. This enables the system to follow individuals over time, even as they move, change pose, or are temporarily occluded.


### How Face Tracking Works

- For each new frame, detected faces are matched to existing tracks based on spatial proximity, appearance features, or motion prediction.
- Each track is assigned a unique `track_id` and maintains a status object that stores information such as the current identity (if recognized), confirmation state, last and next re-identification frames, and other relevant attributes (e.g., database ID, alert status).
- New detections that do not match any existing track start a new track; tracks with no matching detection for several frames are removed.

#### 1. Association
- The system computes a cost (distance) between each detection and existing track (e.g., using bounding box overlap, embedding similarity, or a combination).
- The best matches are assigned using algorithms like Hungarian matching or greedy assignment.

#### 2. Track Management
- Tracks are updated with new detections and their status is refreshed (e.g., confirmation count, last seen frame).
- Tracks that are not matched for a configurable number of frames are deleted (track expiration).

#### 3. Handling Occlusion and Re-Entry
- If a face disappears (e.g., occluded), its track is kept alive for a few frames in case it reappears.
- Advanced trackers may use appearance features to re-link tracks after longer gaps.


### Why Use Face Tracking?

- **Identity Persistence:** Maintains consistent IDs for faces across frames.
- **Event Detection:** Enables logic like "person entered/exited" or "person seen multiple times."
- **Status Tracking:** Each track's status object keeps track of identity, confirmation, and alerting information, supporting downstream logic.
- **Efficiency:** Reduces redundant processing by linking detections over time.

### Example

A person walks across the camera view. The tracker assigns them `track_id=5` and updates their position in each frame. If they leave and return, a new track may be created, or the system may re-link to the old track if appearance features match.

> **Tip:** Tune the association and track expiration parameters for your scenario. For crowded scenes, appearance-based tracking is more robust than position-only tracking.


## Understanding the Face Filters System

The **Face Filters System** is a modular set of filters that operate on detected faces in a video stream or image sequence. These filters help ensure that only faces meeting certain quality or spatial criteria are passed downstream for recognition, tracking, or display.

### Why Use Face Filters?

- **Quality Control:** Remove faces that are too small, poorly aligned, or not frontal enough for reliable recognition.
- **Performance:** Reduce computational load by discarding detections unlikely to be useful.
- **Application Logic:** Enforce business rules, such as only recognizing faces in a specific region of interest (ROI).

### Types of Face Filters

The system typically includes the following filters (see `face_filters.py`):

#### 1. Landmarks Filter
- **Purpose:** Ensures that a face detection has a valid set of facial landmarks (eyes, nose, mouth, etc.).
- **Logic:** If landmarks are missing or invalid, the detection is filtered out.

#### 2. Size Filter
- **Purpose:** Removes faces that are too small or too large for reliable recognition.
- **Parameters:** `min_size`, `max_size` (in pixels or relative to frame size).
- **Logic:** If the bounding box size is outside the allowed range, the detection is filtered out.

#### 3. Zone Filter
- **Purpose:** Restricts recognition to faces within a specific region of the frame (e.g., a door area).
- **Parameters:** `zone` (polygon or rectangle coordinates).
- **Logic:** If the face center is outside the zone, the detection is filtered out.

#### 4. Frontalness Filter
- **Purpose:** Ensures that only faces looking toward the camera are processed.
- **Parameters:** `min_frontalness` (threshold for how "frontal" the face must be).
- **Logic:** Uses pose estimation or landmark symmetry to estimate frontalness; filters out faces below the threshold.

#### 5. Shift Filter
- **Purpose:** Removes faces that are not well-centered in their bounding box (e.g., due to partial occlusion or poor detection).
- **Parameters:** `max_shift` (maximum allowed offset of landmarks from box center).
- **Logic:** If the face is shifted too far from the box center, it is filtered out.

### How Filters Are Applied

Filters are typically applied in sequence. A face must pass all filters to be accepted. The filter pipeline can be customized by enabling/disabling filters or adjusting their parameters.

#### Example Filter Pipeline

```python
filters = [
    LandmarksFilter(),
    SizeFilter(min_size=40, max_size=400),
    ZoneFilter(zone=roi_polygon),
    FrontalnessFilter(min_frontalness=0.7),
    ShiftFilter(max_shift=0.2),
]

for face in detected_faces:
    if all(f(face) for f in filters):
        process(face)
```

### Customizing Filters

- **Parameters:** Tune thresholds (e.g., `min_size`, `min_frontalness`) for your application and camera setup.
- **Order:** Place stricter or cheaper filters first for efficiency.
- **Extensibility:** Implement custom filters by subclassing the base filter class and adding them to the pipeline.

### Debugging and Visualization

- **Debug Output:** Many pipelines support debug overlays to show which faces were filtered out and why.
- **Logs:** Filter decisions can be logged for analysis and tuning.

### Typical Use Case

1. Detect faces in each frame.
2. Apply the filter pipeline to each detection.
3. Only pass faces that pass all filters to the recognition/tracking system.
4. Optionally, visualize or log filtered-out faces for review.

## Understanding reID Scheduling Logic

### What is reID Scheduling?

In face tracking, **reID** (re-identification) is the process of running the face embedding model on a detected face to generate a feature vector (embedding), which is then used to match the face to known identities or confirm it as unknown. To optimize performance and avoid redundant computations, the system does not re-identify every face in every frame. Instead, it uses a scheduling logic to determine when a face should be re-identified (i.e., when a new embedding should be computed).

### How reID Scheduling Works

- Each detected face is assigned a `track_id` and tracked across frames.

##### 2. Expiration Window

- The parameter `reid_expiration_frames` controls how many frames a face can go without being re-identified (embedding updated).
- Each face status stores:
  - `last_reid_frame`: The last frame number when reID (embedding model run) was performed.
  - `next_reid_frame`: The next frame number when reID should be performed.

##### 3. Scheduling Logic

- For each face in a new frame:
  - If the face is new (not in the map), it is immediately scheduled for reID (embedding model is run).
  - If the face is already tracked:
    - If the current frame number is less than `next_reid_frame`, reID is skipped for this face (cached embedding is reused).
    - If the current frame number is equal to or greater than `next_reid_frame`, reID is performed (embedding model is run), and the scheduling window is updated.

##### 4. Exponential Backoff Strategy

- After each successful reID, the scheduling window is recalculated using an **exponential backoff strategy**:
  - The next scheduled reID is set to the current frame plus a delta.
  - Delta is the minimum of `reid_expiration_frames` and twice the previous interval, allowing the window to adapt to face stability and reduce unnecessary reID calls for faces that remain stable over time.

##### 5. Cleanup

- Faces that have not been seen for more than `reid_expiration_frames` are removed from the map to free resources.

##### 6. Rapid Reconfirmation on Identity Change

- When a tracked face is re-identified and its database ID (`db_id`) changes (i.e., the system thinks it is now a different person or status):
  - The confirmation count is reset to 1.
  - The `next_reid_frame` is set to `last_reid_frame + 1`, forcing the system to attempt reID (run the embedding model) again in the very next frame.
  - This allows for rapid reconfirmation and quick correction if a face's identity changes, making the system responsive to identity switches.

**Example code:**

```python
if face.db_id == db_id:
    face.confirmed_count += 1
else:
    face.confirmed_count = 1
    # reset frame counter when the face changes status for quick reconfirming
    face.next_reid_frame = face.last_reid_frame + 1
```

### Why Use reID Scheduling?

- **Efficiency:** Reduces unnecessary runs of the embedding model for faces that are continuously tracked.
- **Responsiveness:** Ensures that faces are re-identified often enough to catch changes (e.g., a person leaves and returns, or identity changes).
- **Resource Management:** Automatically cleans up old face tracks that are no longer present.
- **Exponential Backoff:** The strategy adapts the reID frequency based on face stability, reducing computation for stable faces while remaining responsive to changes.

### Example


Suppose `reid_expiration_frames` is set to 10:

- A face is detected and re-identified (embedding model is run) at frame 100 (`last_reid_frame = 100`, `next_reid_frame = 101`).
  - If the face remains tracked, the interval before the next reID may double, up to the maximum set by `reid_expiration_frames`.
  - If the face's identity changes, the system will attempt to re-identify it again in the very next frame for rapid confirmation.

> **Tip:** Adjust `reid_expiration_frames` based on your application's needs for accuracy and performance. Lower values mean more frequent reID (higher accuracy, more computation); higher values mean less frequent reID (lower computation, possible delay in identity updates). The exponential backoff and rapid reconfirmation logic ensure that identity changes are handled quickly and reliably, while stable faces are processed efficiently.

## Understanding Face Alignment & Extraction

### What is Face Alignment & Extraction?

Face alignment and extraction is the process of normalizing the orientation and scale of detected faces and cropping them to a standard format. This step ensures that faces are consistently positioned and sized before feature extraction (embedding), improving recognition accuracy.

### How Face Alignment & Extraction Works

- For each detected face, facial landmarks (e.g., eyes, nose, mouth) are detected.
- The landmarks are used to estimate the face's pose and apply a geometric transformation (rotation, scaling, translation) to align the face to a canonical position.
- The aligned face region is then cropped to a fixed size, producing a standardized face image for embedding.

#### 1. Landmark Detection
- Landmarks are detected using a separate model or as part of the face detector.
- Typical landmarks: left eye, right eye, nose tip, mouth corners, chin, etc.

#### 2. Alignment Transformation
- The transformation aligns the eyes and mouth to predefined locations in the output image.
- Common methods: similarity transform, affine transform, or piecewise warping.

#### 3. Cropping & Resizing
- The aligned face is cropped to a fixed size (e.g., 112x112 or 160x160 pixels).
- This ensures all faces have the same input format for the embedding model.

### Why Use Alignment & Extraction?

- **Accuracy:** Reduces variation due to pose, scale, and rotation, improving recognition performance.
- **Consistency:** Standardizes face crops for downstream models.
- **Robustness:** Helps the system handle faces at different angles and positions.

### Example

Suppose a detected face is tilted 20° to the left. After alignment, the eyes and mouth are horizontally aligned, and the face is cropped to a standard size. This aligned crop is then passed to the embedding model.

> **Tip:** If your application only encounters frontal faces, alignment may be less critical. For unconstrained environments, alignment is highly recommended.

## Understanding Face Embedding (Feature Extraction)

### What is Face Embedding?

Face embedding is the process of converting an aligned face image into a compact numerical representation (embedding vector) that captures the unique features of the face. This vector is used for comparing, recognizing, or clustering faces.

### How Face Embedding Works

- An aligned face crop is passed through a deep neural network (embedding model).
- The model outputs a fixed-length vector (e.g., 128 or 512 dimensions) representing the face's features.
- Similar faces produce similar embeddings; different faces produce distant embeddings.

#### 1. Model Selection
- Choose an embedding model based on accuracy, speed, and hardware (e.g., ArcFace, FaceNet, MobileFaceNet).
- Pretrained models are available, or you can train your own on a large face dataset.

#### 2. Preprocessing
- Input face crops are normalized (pixel values, mean subtraction, etc.).
- The input size must match the model's requirements (e.g., 112x112 pixels).

#### 3. Embedding Extraction
- The model processes the input and outputs the embedding vector.
- Embeddings are typically L2-normalized for comparison.

### Why Use Face Embedding?

- **Comparison:** Enables fast, accurate face matching using distance metrics (e.g., cosine, Euclidean).
- **Scalability:** Supports large databases and real-time search.
- **Versatility:** Embeddings can be used for recognition, clustering, or verification.

### Example

Two aligned face crops are passed through the embedding model, producing two vectors. The cosine similarity between the vectors indicates how likely the faces are to be the same person.

> **Tip:** Use L2-normalized embeddings for best results with cosine similarity. Regularly evaluate your embedding model on your data to ensure good performance.



## Understanding Identity Confirmation Logic

### What is Identity Confirmation?

In face tracking, **identity confirmation** is the process of ensuring that a detected face is consistently recognized as the same identity over multiple frames before the system "trusts" the result. This helps prevent flickering labels and false alerts due to occasional misidentification.

### How Identity Confirmation Works

#### 1. Confirmation Count

- Each tracked face maintains a `confirmed_count`—the number of consecutive frames in which the face has been recognized as the same identity (`db_id`).
- When a face is recognized with the same `db_id` as before, `confirmed_count` is incremented.
- If the `db_id` changes (the system thinks the face is now a different person), `confirmed_count` is reset to 1, and rapid reconfirmation is triggered (see reID expiration logic).

#### 2. Confirmation Threshold

- The system uses a parameter (e.g., `credence_count`) to determine how many consistent recognitions are required before a face is considered "confirmed."
- Only after `confirmed_count` reaches this threshold is the face marked as `is_confirmed = True`.

#### 3. Why Identity Confirmation Matters

- **Stability:** Prevents identity flicker by requiring agreement across multiple frames.
- **Accuracy:** Reduces the chance of false positives from a single misclassification.
- **Alerting:** Ensures alerts are only sent when a face is reliably identified.

#### 4. Example Logic

```python
if face.db_id == db_id:
    face.confirmed_count += 1
else:
    face.confirmed_count = 1
    face.next_reid_frame = face.last_reid_frame + 1  # rapid reconfirmation

face.is_confirmed = face.confirmed_count >= credence_count
```


#### 5. Alerting on Identity Confirmation

- Once a face is confirmed, the system can trigger alerts (e.g., "Person X seen") and mark the face as alerted to avoid duplicate notifications.
- If the face's identity or attributes change, the alert status can be reset (if configured).

### Example

Suppose `credence_count` is set to 3:

- A face is detected and recognized as "Alice" in three consecutive frames.
- After the third consistent recognition, `confirmed_count` reaches 3, and the face is marked as confirmed.
- If the identity changes to "Bob," `confirmed_count` resets to 1, and the confirmation process starts over.

> **Tip:** Adjust `credence_count` based on your application's tolerance for false positives and the expected video quality. Lower values confirm faces faster but may be less robust; higher values are more conservative but may delay confirmation.

## Understanding the Annotation Stage


### What is Annotation?


Annotation is the stage in the face recognition pipeline where human-readable labels, attributes, and metadata are attached to each detected face. This makes the results interpretable for users and downstream systems, enabling visualization, alerting, and logging.


### How Annotation Works


- For each detected face, the system determines a label based on tracking and confirmation status (e.g., "identifying", "confirming", known identity, or "UNKNOWN").
- Attributes such as identity, confidence, and custom metadata are attached to each face.
- The annotation logic can be customized using a label map or application-specific rules.
- The enriched results are passed to downstream consumers (UI overlays, alerting modules, logs, etc.).


#### 1. Label Assignment
- If a face is not tracked, it is labeled as "not tracked".
- If tracked but not yet in the object map, it is labeled as "identifying".
- If confirmed and has known attributes, the label is set to the attribute string.
- If confirmed but has no attributes, it is labeled as "UNKNOWN".
- If not yet confirmed, it is labeled as "confirming".


#### 2. Customization
- The label assignment logic can be customized using a `label_map` or by extending the annotation logic for your application.


#### 3. Output
- The annotated results include labels, attributes, and any additional metadata, ready for visualization, alerting, or storage.


### Why Use Annotation?


- **Interpretability:** Provides clear, human-readable labels for each face, reflecting its current tracking and confirmation state.
- **Debugging:** Makes it easy to see which faces are being tracked, which are still being confirmed, and which are unknown.
- **Integration:** Downstream systems (e.g., UI overlays, alerting modules) can use these labels to display or act on face statuses.

### Example Labeling Logic

```python
if track_id is None:
    label = "not tracked"
elif obj_status is None:
    label = "identifying"
elif obj_status.is_confirmed:
    if obj_status.attributes is None:
        label = "UNKNOWN"
    else:
        label = str(obj_status)
else:
    label = "confirming"
```

### Typical Use Case

- After face detection, tracking, and confirmation, the annotation stage assigns labels and attributes before results are displayed, logged, or used for alerts.
- This ensures that only confirmed identities are shown as known, while new or unconfirmed faces are clearly marked.

# Understanding the Alerting Stage

## What is Alerting?

Alerting is the stage in the face recognition pipeline that triggers notifications, logs, or actions when certain face events occur—such as a known or unknown person being confirmed. It enables real-time responses to security or business events.

## How Alerting Works

At each frame, the system checks the status of every tracked face. When a face is confirmed (i.e., recognized as the same identity for enough frames), the system decides whether to trigger an alert based on configurable rules:

- **Alert on unknown faces:** Trigger when a confirmed face has no attributes (not recognized in the database).
- **Alert on known faces:** Trigger when a confirmed face has known attributes (recognized identity).
- **Alert on all faces:** Trigger for any confirmed face, regardless of identity.

You can choose to alert only once per face, or allow repeated alerts if the face's identity or attributes change.

### Example Decision Flow

For each face:

1. If the face is confirmed:
    - If alerting on unknowns and the face is unknown, trigger alert.
    - If alerting on knowns and the face is known, trigger alert.
    - If alerting on all, trigger alert.
2. If the face has already triggered an alert and repeated alerts are not allowed, do not trigger again unless the face's attributes change.

**Example logic:**

```python
if is_confirmed:
    if (
        (alert_on == "unknowns" and attributes is None and not is_alerted)
        or (alert_on == "knowns" and attributes is not None and not is_alerted)
        or (alert_on == "all" and not is_alerted)
    ):
        trigger_alert()
        is_alerted = True
```

Alerts can be handled by downstream logic, such as sending notifications, logging, or triggering automated actions.

If repeated alerts are allowed, the alert status can be reset when the face's attributes change.

## Why Use Alerting?

- **Real-Time Response:** Enables immediate action when important events occur (e.g., unauthorized entry).
- **Security:** Notifies staff or systems of unknown or specific known individuals.
- **Automation:** Integrates with external systems for automated responses (e.g., open doors, trigger alarms).

## Example: Configuring and Using Alerting

Suppose you want to alert only on unknown faces, and only once per face:

```python
alert_on = "unknowns"  # or "knowns", "all"
alert_once = True

for face in faces:
    if face.is_confirmed:
        if (
            (alert_on == "unknowns" and face.attributes is None and not face.is_alerted)
            or (alert_on == "knowns" and face.attributes is not None and not face.is_alerted)
            or (alert_on == "all" and not face.is_alerted)
        ):
            send_alert_notification(face)
            face.is_alerted = True
```

## Typical Use Case

1. After annotation and confirmation, the alerting stage checks each face for alert conditions.
2. If a condition is met, an alert is triggered and can be consumed by downstream logic (UI, logs, notifications).
3. Alerts are deduplicated per face unless configured otherwise.


## Tips and Best Practices

- Choose the appropriate alerting rule for your application (e.g., only alert on unknowns for security, or on all faces for analytics).
- Use the `alert_once` flag to control whether repeated alerts are allowed for the same face.
- Integrate alert handling with your notification, logging, or automation systems.
- Monitor and log alert events for auditing and tuning your alerting rules.


## Understanding the Recording and Storing Stage

### What is Recording and Storing?

Recording and Storing is the stage in the face recognition pipeline where video clips, images, or metadata are saved when important events occur (such as an unknown or known person being detected). This enables later review, annotation, and evidence collection.

### How Recording and Storing Works

- When an alert is triggered, the system saves relevant video clips or images to a configured storage location (local disk or object storage).
- Each saved clip is associated with metadata such as time, event type, and (optionally) recognized identity.
- Saved clips can be listed, reviewed, annotated, and deleted through the application interface.
- Annotations and new identities can be added to clips after review, and the face database can be updated with new embeddings from annotated clips.

#### Example Flow

1. An alert event occurs (e.g., unknown person detected).
2. The system records a video clip and saves it to storage with associated metadata.
3. Users can review and annotate clips, updating identities or adding new ones as needed.
4. Embeddings from annotated clips can be added to the face database to improve future recognition.

### Why Use Recording and Storing?

- **Audit Trail:** Maintains a record of important events for later review.
- **Security:** Supports investigation of incidents involving known or unknown individuals.
- **Continuous Improvement:** Enables updating the face database with new identities and embeddings from real-world events.

### Example Recording Logic

```python
if alert_event:
    save_clip_to_storage(video_segment, metadata)
```

### Typical Use Case

- When an alert is triggered, a video clip is saved and can later be reviewed and annotated.
- The system supports workflows for managing, annotating, and updating the face database using stored clips.

---
