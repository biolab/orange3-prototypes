import numpy as np

import pandas as pd
from pandas.api.types import (
    is_categorical_dtype, is_object_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from Orange.data import (
    Table, Domain, DiscreteVariable, ContinuousVariable,
    StringVariable, TimeVariable,
)


def table_from_frame(df, *, force_nominal=False):

    def _is_discrete(s):
        return (is_categorical_dtype(s) or
                is_object_dtype(s) and (force_nominal or
                                        s.nunique() < s.size**.666))

    def _is_datetime(s):
        if is_datetime64_any_dtype(s):
            return True
        try:
            if is_object_dtype(s):
                pd.to_datetime(s, infer_datetime_format=True)
                return True
        except Exception:
            pass
        return False

    attrs, metas = [], []
    X, M = [], []

    for name, s in df.items():
        name = str(name)
        if _is_discrete(s):
            discrete = s.astype('category').cat
            attrs.append(DiscreteVariable(name, discrete.categories.astype(str).tolist()))
            X.append(discrete.codes.replace(-1, np.nan).values)
        elif _is_datetime(s):
            tvar = TimeVariable(name)
            attrs.append(tvar)
            s = pd.to_datetime(s, infer_datetime_format=True)
            X.append(s.astype('str').map(tvar.parse).values)
        elif is_numeric_dtype(s):
            attrs.append(ContinuousVariable(name))
            X.append(s.values)
        else:
            metas.append(StringVariable(name))
            M.append(s.values.astype(object))

    MAX_LENGTH = max(len(X[0]) if X else 0,
                     len(M[0]) if M else 0)
    return Table.from_numpy(Domain(attrs, None, metas),
                            np.column_stack(X) if X else np.empty(
                                (MAX_LENGTH, 0)),
                            None, np.column_stack(M) if M else None)
