from dataclasses import dataclass

from dataclasses_json import dataclass_json

from model.HintType import HintType


@dataclass_json
@dataclass
class Hint:
    """「サングラス」のヒントデータ"""
    type: HintType  # "row"か"col"。rowだと行における個数を規定し、colだと列における個数を規定する
    index: int      # インデックス。例えば"row"でindex=2なら、3行目であることを示す
    value: int      # 個数値。この数字の分だけ、レンズのために塗りつぶすことができる
