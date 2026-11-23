"""Qdrant: two collections (frames, transcript), one per everyone, filtered by owner."""
import uuid

from .core import S, USER_ID, IMAGE_COLLECTION, TEXT_COLLECTION, SPARSE_VECTOR, \
    QDRANT_ON_DISK, QDRANT_HNSW_ON_DISK, QDRANT_QUANTIZATION


def create_collection(name, dim, sparse=False):
    from qdrant_client import models as qm
    S.qdrant.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE, on_disk=QDRANT_ON_DISK),
        hnsw_config=qm.HnswConfigDiff(on_disk=QDRANT_HNSW_ON_DISK),
        quantization_config=(qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(
            type=qm.ScalarType.INT8, always_ram=True)) if QDRANT_QUANTIZATION else None),
        sparse_vectors_config={SPARSE_VECTOR: qm.SparseVectorParams(modifier=qm.Modifier.IDF)} if sparse else None,
    )
    S.qdrant.create_payload_index(name, "user_id", field_schema=qm.KeywordIndexParams(
        type=qm.KeywordIndexType.KEYWORD, is_tenant=True))
    S.qdrant.create_payload_index(name, "video_id", field_schema=qm.PayloadSchemaType.KEYWORD)


def ensure_collection(name, dim, sparse=False):
    if not S.qdrant.collection_exists(name):
        create_collection(name, dim, sparse)
        return
    found = getattr(S.qdrant.get_collection(name).config.params.vectors, "size", None)
    if found and found != dim:
        raise RuntimeError(f"'{name}' holds {found}-number vectors but the model makes {dim}.")


def point_id(vid, idx):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{vid}:{idx}"))


def chunk_point_id(vid, i):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{vid}:text:{i}"))


def owner_filter(video_ids=None, user_id=None):
    """Every search and delete goes through this: only this user's points, optionally only some videos."""
    from qdrant_client import models as qm
    must = [qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id or USER_ID))]
    if video_ids:
        must.append(qm.FieldCondition(key="video_id", match=qm.MatchAny(any=list(video_ids))))
    return qm.Filter(must=must)


def delete_video(vid):
    from qdrant_client import models as qm
    for coll in (IMAGE_COLLECTION, TEXT_COLLECTION):
        if S.qdrant.collection_exists(coll):
            S.qdrant.delete(coll, points_selector=qm.FilterSelector(filter=owner_filter([vid])), wait=True)
