from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json

from model.CellPos import CellPos


@dataclass_json
@dataclass
class Bridge:
    """「サングラス」のブリッジデータ"""
    cell: List[CellPos]
