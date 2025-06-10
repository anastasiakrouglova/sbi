#%%
from typing import Optional, Tuple


def z_score_parser(z_score_flag: Optional[str]) -> Tuple[bool, bool]:
    """Parses string z-score flag into booleans.

    Converts string flag into booleans denoting whether to z-score or not, and whether
    data dimensions are structured or independent.

    Args:
        z_score_flag: str flag for z-scoring method stating whether the data
            dimensions are "structured" or "independent", or does not require z-scoring
            ("none" or None).

    Returns:
        Flag for whether or not to z-score, and whether data is structured
    """
    if isinstance(z_score_flag, bool):
        # Raise warning if boolean was passed.
        warnings.warn(
            "Boolean flag for z-scoring is deprecated as of sbi v0.18.0. It will be "
            "removed in a future release. Use 'none', 'independent', or 'structured' "
            "to indicate z-scoring option.",
            stacklevel=2,
        )
        z_score_bool, structured_data = z_score_flag, False

    elif (z_score_flag is None) or (z_score_flag == "none"):
        # Return Falses if "none" or None was passed.
        z_score_bool, structured_data = False, False

    elif (z_score_flag == "independent") or (z_score_flag == "structured"):
        # Got one of two valid z-scoring methods.
        z_score_bool = True
        structured_data = z_score_flag == "structured"
    elif z_score_flag == "logit":
        # Do not z-score if logit transform. Logit is not estimated from data,
        # but from the prior bounds, so structured/indpendent does not matter.
        z_score_bool, structured_data = False, False
    else:
        # Return warning due to invalid option, defaults to not z-scoring.
        raise ValueError(
            "Invalid z-scoring option. Use 'none', 'independent', or 'structured'."
        )

    return z_score_bool, structured_data



z_score_parser("none")
# %%
