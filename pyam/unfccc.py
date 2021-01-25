import numpy as np
import unfccc_di_api

from pyam import IamDataFrame
from pyam.utils import pattern_match, isstr

# columns from UNFCCC data that can be used for variable names
NAME_COLS = ['category', 'classification', 'measure', 'gas']

# UNFCCC-reader instance (instantiated at first use)
_READER = None


def read_unfccc(party_code, mapping, gases=None, model='UNFCCC',
                scenario='Data Inventory'):
    """Read data from the UNFCCC Data Inventory

    This function is a wrappter for the
    :meth:`unfccc_di_api.UNFCCCApiReader.query`.

    Parameters
    ----------
    party_code
    mapping
    gases
    model
    scenario

    Returns
    -------

    """
    global _READER
    if _READER is None:
        _READER = unfccc_di_api.UNFCCCApiReader()

    # retrieve data, drop non-numeric data and base year
    data = _READER.query(party_code=party_code, gases=gases)
    data = data[~np.isnan(data.numberValue)]
    data = data[data.year != 'Base year']

    # add new column, iterate over mapping to determine variables
    data['variable'] = None
    for variable, value in mapping.items():
        matches = np.array([True] * len(data))
        for i, col in enumerate(NAME_COLS):
            matches &= pattern_match(data[col], value[i])

        data.loc[matches, 'variable'] = \
            data[matches].apply(compile_variable, variable=variable, axis=1)

    # drop unspecified rows and columns, rename value column
    cols = ['party', 'variable', 'unit', 'year', 'numberValue']
    data = (
        data.loc[[isstr(i) for i in data.variable], cols]
        .rename(columns={'numberValue': 'value'})
    )

    # cast to IamDataFrame
    return IamDataFrame(data, model=model, scenario=scenario, region='party')


def compile_variable(i, variable):
    """Translate UNFCCC columns into an IAMC-style variable"""
    if i['variable']:
        raise ValueError('Conflict in variable mapping!')
    return variable.format(**dict((c, i[c]) for c in NAME_COLS))