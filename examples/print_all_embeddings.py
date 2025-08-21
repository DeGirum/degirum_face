import degirum_face
import numpy as np

# Open the database
# (adjust path if needed)
db = degirum_face.ReID_Database("face_recognition.lance")

# Open the embeddings table
et, _ = db._open_table(degirum_face.ReID_Database.tbl_embeddings)

if et is None:
    print("No embeddings table found.")
    exit(0)

# Query all embeddings and their object IDs
results = (
    et.search()
    .select(
        [
            degirum_face.ReID_Database.key_object_id,
            degirum_face.ReID_Database.key_embedding,
        ]
    )
    .to_list()
)

if not results:
    print("No embeddings found in the database.")
else:
    for i, row in enumerate(results):
        obj_id = row.get(degirum_face.ReID_Database.key_object_id)
        embedding = row.get(degirum_face.ReID_Database.key_embedding)
        print(f"Embedding #{i+1} for object_id {obj_id}:")
        print(np.array(embedding))
        print()
