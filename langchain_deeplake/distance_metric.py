from typing import List, Optional


def cosine_similarity(embedding_tensor, query_embedding):
    return f"COSINE_SIMILARITY({embedding_tensor}, {query_embedding})"


def l2_norm(embedding_tensor, query_embedding):
    return f"L2_NORM({embedding_tensor}-{query_embedding})"


METRIC_TO_TQL_QUERY = {
    "l2": l2_norm,
    "cos": cosine_similarity,
}


def get_tql_distance_metric(metric: str, embedding_tensor: str, query_embedding: str):
    return METRIC_TO_TQL_QUERY[metric](embedding_tensor, query_embedding)


METRIC_TO_ORDER_TYPE = {
    "l2": "ASC",
    "cos": "DESC",
}


def get_order_type_for_distance_metric(distance_metric):
    return METRIC_TO_ORDER_TYPE[distance_metric]
