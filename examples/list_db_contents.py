from degirum_face.reid_database import ReID_Database

# Path to your LanceDB file
db_path = "face_recognition.lance"

db = ReID_Database(db_path)

print("=== All objects in the database ===")
objects = db.list_objects()
for object_id, attributes in objects.items():
    print(f"ID: {object_id} | Name: {attributes}")

print("\n=== Embedding counts per object ===")
counts = db.count_embeddings()
for object_id, (count, attributes) in counts.items():
    print(f"ID: {object_id} | Name: {attributes} | Embedding count: {count}")
