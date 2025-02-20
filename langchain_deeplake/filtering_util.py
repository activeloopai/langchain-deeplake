import deeplake

from typing import Optional, Dict
from langchain_deeplake.exceptions import ColumnMissingError


def attribute_based_filtering_tql(
    ds: deeplake.Dataset, filter: Optional[Dict] = None
) -> str:
    """Filter helper function converting filter dictionary to TQL Deep Lake
    For non-dict tensors, perform exact match if target data is not a list, and perform "IN" match if target data is a list.
    For dict tensors, perform exact match for each key-value pair in the target data.
    """

    tql_filter = ""

    if filter is not None:
        if isinstance(filter, dict):
            columns = ds.schema.columns
            column_names = [column.name for column in columns]
            for column in filter.keys():
                if column not in column_names:
                    raise ColumnMissingError(column)
                if ds.schema[column].dtype.kind == deeplake.types.TypeKind.Dict:
                    for key, value in filter[column].items():
                        val_str = f"'{value}'" if type(value) == str else f"{value}"
                        tql_filter += f"{column}['{key}'] == {val_str} and "
                else:
                    if type(filter[column]) == list:
                        val_str = str(filter[column])[
                            1:-1
                        ]  # Remove square bracked and add rounded brackets below.

                        tql_filter += f"{column} in ({val_str}) and "

                    else:
                        val_str = (
                            f"'{filter[column]}'"
                            if isinstance(filter[column], str)
                            or isinstance(filter[column], np.str_)
                            else f"{filter[column]}"
                        )
                        tql_filter += f"{column} == {val_str} and "

            tql_filter = tql_filter[:-5]

    return tql_filter
