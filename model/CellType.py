from enum import IntEnum


class CellType(IntEnum):
    """マスの種類"""
    UNKNOWN = 0
    LENS = 1
    BLANK = 2


CELL_TYPE_LIST = {
    CellType.UNKNOWN: 'UNKNOWN',
    CellType.LENS: 'LENS',
    CellType.BLANK: 'BLANK'
}
