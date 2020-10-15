from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json

from model.Bridge import Bridge
from model.Hint import Hint


@dataclass_json
@dataclass
class SunGlass:
    """「サングラス」の問題データ"""
    width: int      # 横幅
    height: int     # 縦幅
    bridge: List[Bridge]    # ブリッジの情報
    hint: List[Hint]        # ヒントの情報
