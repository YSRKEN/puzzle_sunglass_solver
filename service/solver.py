from typing import List

from model.CellPos import CellPos
from model.CellType import CellType
from model.SunGlass import SunGlass


def create_board_from_problem(problem: SunGlass) -> List[CellType]:
    """問題データから初期盤面を生成する

    Parameters
    ----------
    problem
        問題データ

    Returns
    -------
        初期盤面
    """

    def get_pos(pos: CellPos) -> int:
        return pos.x + pos.y * problem.width

    # ブリッジの線から、初期のレンズ位置・空白位置を決定させる
    board = [CellType.UNKNOWN] * (problem.width * problem.height)
    for bridge in problem.bridge:
        board[get_pos(bridge.cell[0])] = CellType.LENS
        board[get_pos(bridge.cell[-1])] = CellType.LENS
        for i in range(1, len(bridge.cell) - 1):
            board[get_pos(bridge.cell[i])] = CellType.BLANK
    return board
