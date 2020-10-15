from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class CellPos:
    """座標"""
    x: int  # X座標
    y: int  # Y座標
