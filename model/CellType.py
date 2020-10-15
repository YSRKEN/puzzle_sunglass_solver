from enum import IntEnum


class CellType(IntEnum):
    """マスの種類"""
    UNKNOWN = 0     # 不明
    LENS = 1        # レンズ
    BLANK = 2       # 空白


CELL_TYPE_LIST = {
    CellType.UNKNOWN: 'UNKNOWN',
    CellType.LENS: 'LENS',
    CellType.BLANK: 'BLANK'
}
