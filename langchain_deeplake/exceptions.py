class ColumnMissingError(Exception):
    def __init__(self, column: str):
        super().__init__(f"Column '{column}' is not present in the Vector Store.")


class UnexpectedUDFFilterError(Exception):
    def __init__(self, column: str):
        super().__init__(
            "UDF filter functions are not supported with DeepLake search. Please use TQL filter instead."
        )


class MissingQueryOrTQLError(ValueError):
    """Exception raised when both query and TQL are missing."""

    def __init__(self, message="Either query or tql must be provided."):
        super().__init__(message)


class InvalidQuerySpecificationError(ValueError):
    """Exception raised when either both or neither of query and tql are provided."""

    def __init__(self, message="Exactly one of 'query' or 'tql' must be provided."):
        super().__init__(message)
