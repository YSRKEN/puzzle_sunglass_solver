from typing import List

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
    board = [CellType.UNKNOWN] * (problem.width * problem.height)
    return board
