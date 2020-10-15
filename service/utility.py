from typing import List

from model.CellPos import CellPos
from model.CellType import CellType
from model.HintType import HintType
from model.SunGlass import SunGlass


def int_to_str(number: int) -> str:
    """表示向けのリッチな数字→文字列変換

    Parameters
    ----------
    number
        数字

    Returns
    -------
        数字文字列
    """
    if number < 10:
        return '０１２３４５６７８９'[number]
    else:
        return str(number)


def show_board_data(board: List[CellType], problem: SunGlass) -> None:
    """盤面を可視化する

    Parameters
    ----------
    board
        盤面データ
    problem
        問題データ

    Returns
    -------
        結果を標準出力で返す
    """

    def get_pos(pos: CellPos) -> int:
        return pos.x + pos.y * problem.width

    # 初期表示内容を決定する
    board_str = ['□' for x in board]

    # ブリッジの線を描画する
    for bridge in problem.bridge:
        for i in range(len(bridge.cell)):
            bridge_cell = bridge.cell[i]
            if i < len(bridge.cell) - 1:
                mx, my = bridge.cell[i + 1].x - bridge_cell.x, bridge.cell[i + 1].y - bridge_cell.y
            else:
                mx, my = bridge_cell.x - bridge.cell[i - 1].x, bridge_cell.y - bridge.cell[i - 1].y
            if mx == 0:
                board_str[get_pos(bridge_cell)] = '│'
            elif my == 0:
                board_str[get_pos(bridge_cell)] = '─'
            elif mx * my > 0:
                board_str[get_pos(bridge_cell)] = '＼'
            else:
                board_str[get_pos(bridge_cell)] = '／'

    # レンズと空白を描画する
    mark = '□■・'
    for i in range(len(board)):
        if board[i] == CellType.UNKNOWN:
            continue
        if board[i] == CellType.BLANK and board_str[i] != '□':
            continue
        board_str[i] = mark[board[i]]

    # 最終的に出力する
    print('　　', end='')
    for x in range(problem.width):
        # ヒント数字
        temp = [t for t in problem.hint if t.index == x and t.type == HintType.COLUMN]
        if len(temp) > 0:
            print(int_to_str(temp[0].value), end='')
        else:
            print('　', end='')
    print('')
    print('　┏' + '━' * problem.width + '┓')
    for y in range(problem.height):
        # ヒント数字
        temp = [t for t in problem.hint if t.index == y and t.type == HintType.ROW]
        if len(temp) > 0:
            print(int_to_str(temp[0].value), end='')
        else:
            print('　', end='')

        # 盤面
        begin_index = y * problem.width
        end_index = begin_index + problem.width
        sliced_board = board_str[begin_index:end_index]
        print('┃' + ''.join(sliced_board) + '┃')
    print('　┗' + '━' * problem.width + '┛')
