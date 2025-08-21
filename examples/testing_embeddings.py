import numpy as np
import degirum_face

db = degirum_face.ReID_Database("face_recognition.lance")
print(db.count_embeddings())

et, _ = db._open_table(degirum_face.ReID_Database.tbl_embeddings)

print(et.schema)

embeddings = (
    et.search()
    .select([degirum_face.ReID_Database.key_embedding])
    .to_arrow()[degirum_face.ReID_Database.key_embedding]
    .to_pylist()
)

for embedding in embeddings:

    embedding = np.array(embedding, dtype=np.float32)
    # print(embedding)
    r = (
        et.search(
            embedding, vector_column_name=degirum_face.ReID_Database.key_embedding
        )
        .metric("cosine")
        .distance_range(-db._threshold, db._threshold)
        .limit(1)
        .to_list()
    )

    if len(r) > 0:
        print(r[0]["_distance"])
    else:
        print("No match found")
