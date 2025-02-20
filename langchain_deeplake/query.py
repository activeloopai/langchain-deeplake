import numpy as np
import deeplake

from langchain_deeplake.distance_metric import (
    get_tql_distance_metric,
    get_order_type_for_distance_metric,
)
from langchain_deeplake.filtering_util import attribute_based_filtering_tql
from langchain_deeplake.exceptions import UnexpectedUDFFilterError


from typing import Any, Dict, List, Optional


def create_query_string(
    distance_metric: Optional[str],
    tql_filter: str,
    limit: int,
    order: Optional[str],
    tensor_list: List[str],
) -> str:
    """Function for creating a query string from a distance metric, limit and order.

    Args:
        distance_metric (str): distance metric to compute similarity of the query embedding with dataset's embeddings.
        tql_filter (str): Additional filter using TQL syntax.
        limit (int): number of samples to return after the search.
        order (str): Type of data ordering after computing similarity score. Defaults to "ASC".
        tensor_list (List[str]): List of tensors to return data for.


    Returns:
        str: TQL representation of the query string.
    """
    tql_filter_str = tql_filter if tql_filter == "" else " where " + tql_filter
    tensor_list_str = ", ".join(tensor_list)
    distance_metric_str = (
        "" if distance_metric is None else f", {distance_metric} as score"
    )

    order_str = "" if order is None else f" order by {distance_metric} {order}"
    score_str = "" if order is None else f", score"

    return f"select {tensor_list_str}{score_str} from (select *{distance_metric_str}{tql_filter_str}{order_str} limit {limit})"


def create_query(
    distance_metric: str,
    embedding_tensor: str,
    query_embedding: str,
    tql_filter: str,
    limit: int,
    tensor_list: List[str],
):
    """Function for creating a query string from a distance metric, embeddings, query_embedding, and limit.

    Args:
        distance_metric (str): distance metric to compute similarity of the query embedding with dataset's embeddings.
        embedding_tensor (str): name of the tensor in the dataset with ``htype = "embedding"``.
        query_embedding (str): embedding representation of the query string converted to str.
        tql_filter (str): Additional filter using TQL syntax.
        limit (int): number of samples to return after the search.
        tensor_list (List[str]): List of tensors to return data for.


    Returns:
        str: TQL representation of the query string.
    """
    order = get_order_type_for_distance_metric(distance_metric)

    tql_distrance_metric = get_tql_distance_metric(
        distance_metric, embedding_tensor, query_embedding
    )

    query = create_query_string(
        tql_distrance_metric, tql_filter, limit, order, tensor_list
    )
    return query


def convert_tensor_to_str(query_embedding: np.ndarray):
    """Function for converting a query embedding to a string.

    Args:
        query_embedding (Union[List[float], np.ndarray]) - embedding representation of the query string.
    """
    query_embedding_str = ", ".join([str(e) for e in query_embedding])
    return f"ARRAY[{query_embedding_str[:-2]}]"


def search(
    dataset: deeplake.Dataset,
    query_embedding: np.ndarray,
    distance_metric: str,
    k: int,
    tql_string: str,
    tql_filter: Optional[Dict] = None,
    embedding_tensor: str = "embeddings",
    return_tensors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generalized search algorithm that uses indra. It combines vector search and other TQL queries.

    Args:
        dataset (DeepLakeDataset): DeepLake dataset object.
        query_embedding (Optional[Union[List[float], np.ndarray): embedding representation of the query.
        distance_metric (str): Distance metric to compute similarity between query embedding and dataset embeddings
        k (int): number of samples to return after the search.
        tql_string (str): Standalone TQL query for execution without other filters.
        tql_filter (str): Additional filter using TQL syntax
        embedding_tensor (str): name of the tensor in the dataset with `htype = "embedding"`.
        runtime (dict): Runtime parameters for the query.
        return_tensors (List[str]): List of tensors to return data for.

    Raises:
        UnexpectedUDFFilterError: If both tql_string and tql_filter are specified.

    Returns:
        Dict: Dictionary where keys are tensor names and values are the results of the search
    """

    if callable(tql_filter):
        raise UnexpectedUDFFilterError()

    tql_filter = attribute_based_filtering_tql(ds=dataset, filter=tql_filter)

    if tql_string:
        return tql_string

    if query_embedding is None:
        return create_query_string(
            None, tql_filter, limit, None, tensor_list=return_tensors
        )
    else:
        query_embedding_str = convert_tensor_to_str(query_embedding)

        return create_query(
            distance_metric=distance_metric.lower(),
            embedding_tensor=embedding_tensor,
            query_embedding=query_embedding_str,
            tql_filter=tql_filter,
            limit=k,
            tensor_list=return_tensors,
        )
