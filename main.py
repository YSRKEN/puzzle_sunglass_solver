import json
from typing import List

from model.CellType import CellType
from model.SunGlass import SunGlass
from service.solver import create_board_from_problem
from service.utility import show_board_data


def solve(problem: SunGlass) -> None:
    """問題データから計算を行い、解答を標準出力で返す

    Parameters
    ----------
    problem
        問題データ

    Returns
    -------
        標準出力で処理結果を返す
    """

    # 初期盤面データを作成する
    board = create_board_from_problem(problem)
    show_board_data(board, problem)


def solve_from_path(path: str) -> None:
    """問題ファイルをファイルパスから読み込んで処理する

    Parameters
    ----------
    path
        ファイルパス

    Returns
    -------
        標準出力で処理結果を返す
    """
    with open(path) as f:
        problem = SunGlass.from_dict(json.load(f))

    solve(problem)


if __name__ == '__main__':
    solve_from_path('sample_problem.json')
    # solve_from_path('problem/fuwa_lica_chan-1314913582467883009.json')
    # solve_from_path('problem/nyoroppyi-1206580414657089537.json')
