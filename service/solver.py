import json
from typing import List

from model.CellPos import CellPos
from model.CellType import CellType
from model.SunGlass import SunGlass
from service.utility import show_board_data


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


def find_lens(board: List[CellType], width: int, pos: int, taboo_list: List[int] = None) -> List[int]:
    """その座標を含む、レンズ群を取得する

        Parameters
        ----------
        board
            盤面
        width
            盤面の横幅
        pos
            座標
        taboo_list
            タブーリスト。ここに含まれているものは再度カウントしない

        Returns
        -------
            レンズ群における座標の一覧
        """

    # タブーリストを初期化
    if taboo_list is None:
        taboo_list = [pos]

    # 処理開始。現在位置から、上下左右に進んだ位置のマスについて、再帰を行い、結果をmergeする
    output = [pos]
    for offset in (-width, 1, width, -1):
        # 盤面からはみ出さないように判定
        if offset == -width and pos < width:
            continue
        if offset == 1 and (pos + 1) % width == 0:
            continue
        if offset == width and pos + width >= len(board):
            continue
        if offset == -1 and pos % width == 0:
            continue

        next_pos = pos + offset
        # 次の位置にレンズが無かった場合は無視する
        if board[next_pos] != CellType.LENS:
            continue

        # 次の位置がタブーリストに含まれている場合は無視する
        if next_pos in taboo_list:
            continue

        # 再帰した結果を追加
        output += find_lens(board, width, next_pos, taboo_list + [next_pos])
    return output


def pattern_do_not_join_lenses(board: List[CellType], problem: SunGlass) -> List[CellType]:
    """レンズ同士が接触しないように空白を設置

    Parameters
    ----------
    board
        盤面
    problem
        問題

    Returns
    -------
        処理後の盤面
    """

    # タイリングを実施。lens_mapは、各タイル(＝各レンズ)がある位置は、そのタイルの番号(lens_index)が振られるようにする
    lens_map = [0 for _ in board]
    lens_index = 1
    for bridge in problem.bridge:
        # ブリッジに対して左翼のレンズの処理
        for pos in find_lens(board, problem.width, bridge.cell[0].x + bridge.cell[0].y * problem.width):
            lens_map[pos] = lens_index
        lens_index += 1

        # ブリッジに対して右翼のレンズの処理
        for pos in find_lens(board, problem.width, bridge.cell[-1].x + bridge.cell[-1].y * problem.width):
            lens_map[pos] = lens_index
        lens_index += 1

    # 空白を設置
    output = [x for x in board]
    for y in range(problem.height):
        for x in range(problem.width):
            # 指定したマスにおける上下左右のマスの合法な座標一覧
            pos_list = [(x + i) + (y + j) * problem.width
                        for i, j
                        in ((0, -1), (1, 0), (0, 1), (-1, 0))
                        if 0 <= (x + i) < problem.width and 0 <= (y + j) < problem.height]

            # 上下左右のマスのレンズ番号を収集し、setで重複を排除する
            around_lens_set = {lens_map[k] for k in pos_list if lens_map[k] > 0}

            # この判定式が真＝上下左右の周囲に2種類以上のタイルがある
            if len(around_lens_set) > 1:
                output[x + y * problem.width] = CellType.BLANK
    return output


def is_equal(board1: List[CellType], board2: List[CellType]) -> bool:
    """2つの盤面が等しければTrue

    Parameters
    ----------
    board1
        盤面1
    board2
        盤面2

    Returns
    -------
        等しければTrue
    """
    for a, b in zip(board1, board2):
        if a != b:
            return False
    return True


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

    # 各種定石を適用する
    while True:
        # レンズ同士が接触しないように空白を設置
        next_board = pattern_do_not_join_lenses(board, problem)
        if not is_equal(board, next_board):
            board = next_board
            show_board_data(board, problem)
            continue
        break


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
