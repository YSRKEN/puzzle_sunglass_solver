import json
from functools import lru_cache
from typing import List, Tuple, Set

from model.CellPos import CellPos
from model.CellType import CellType
from model.HintType import HintType
from model.SunGlass import SunGlass
from service.utility import show_board_data


class AlgorithmException(Exception):
    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


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


@lru_cache(maxsize=None)
def get_next_pos(pos: int, width: int, board_size: int) -> List[int]:
    """次の座標の一覧を割り出す

    Parameters
    ----------
    pos
        座標
    width
        横幅
    board_size
        全マス数

    Returns
    -------
        次に進めるマスの一覧
    """
    output: List[int] = []
    for offset in (-width, 1, width, -1):
        # 盤面からはみ出さないように判定
        if offset == -width and pos < width:
            continue
        if offset == 1 and (pos + 1) % width == 0:
            continue
        if offset == width and pos + width >= board_size:
            continue
        if offset == -1 and pos % width == 0:
            continue
        output.append(pos + offset)
    return output


def find_lens_max(board: List[CellType], width: int, board_size: int, lens_set: Set[int],
                  other_lens_set: Set[int]) -> List[int]:
    # 前処理により、レンズにならない箇所をあらかじめ空白マスにしておいた盤面を用意する
    board2 = board.copy()
    for i in range(board_size):
        if board[i] == CellType.LENS and i in other_lens_set:
            around_pos_list = get_next_pos(i, width, board_size)
            for pos2 in around_pos_list:
                if board[pos2] == CellType.UNKNOWN:
                    board2[pos2] = CellType.BLANK

    """
    print([pos_int_to_tuple(t, width) for t in lens_set])
    mark = '?o.'
    for i in range(board_size):
        print(f'{mark[int(board2[i])]} ', end='')
        if i % width == width - 1:
            print('')
    print('')
    """

    # レンズになりうる座標一覧をsetで管理し、幅優先探索で順繰りに更新していく
    output = set() | lens_set
    frontier = lens_set.copy()
    # print(f'lens={lens_set}')
    while True:
        # print(f'  frontier={frontier} output={output}')
        new_frontier = set()
        for pos2 in frontier:
            new_frontier = new_frontier | set(get_next_pos(pos2, width, board_size))
        new_frontier = new_frontier - frontier
        new_frontier = {x for x in new_frontier if board2[x] != CellType.BLANK}
        # print(f'  new_frontier={new_frontier}')
        if len(new_frontier) == 0:
            break
        output = output | new_frontier
        frontier = frontier | new_frontier
    """
    print('frontier')
    for i in range(board_size):
        if i in frontier:
            print(f'o ', end='')
        else:
            print(f'. ', end='')
        if i % width == width - 1:
            print('')
    print('')
    """

    return list(frontier)


def calc_lens_map(board: List[CellType], problem: SunGlass) -> List[int]:
    """タイリングを実施

    Parameters
    ----------
    board
        盤面
    problem
        問題

    Returns
    -------
        各タイル(＝ブリッジから生えた各レンズ)がある位置は、そのタイルの番号が振られるようにする。
        タイルの番号は1スタートの自然数であり、ブリッジから生えてない孤立レンズも認識されない
    """
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
    return  lens_map


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
    lens_map = calc_lens_map(board, problem)

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


def find_around_blank_cells(board: List[CellType], width: int, height: int, lens: List[Tuple[int, int]])\
        -> List[Tuple[int, int]]:
    """レンズの周囲における、「レンズの周囲の空白マス」の一覧を取得する

    Parameters
    ----------
    board
        盤面
    width
        横幅
    height
        縦幅
    lens
        レンズ

    Returns
    -------
        「レンズの周囲の空白マス」の一覧
    """
    # noinspection PyTypeChecker
    output: Set[Tuple[int, int]] = set()
    for x, y in lens:
        for pos in get_next_pos(x + y * width, width, width * height):
            if board[pos] != CellType.BLANK:
                continue

            output.add((pos % width, pos // width))
    return list(output)


def find_around_unknown_cells(board: List[CellType], width: int, height: int, lens: List[Tuple[int, int]])\
        -> List[Tuple[int, int]]:
    """レンズの周囲における、「レンズの周囲の不明マス」の一覧を取得する

    Parameters
    ----------
    board
        盤面
    width
        横幅
    height
        縦幅
    lens
        レンズ

    Returns
    -------
        「レンズの周囲の不明マス」の一覧
    """
    # noinspection PyTypeChecker
    output: Set[Tuple[int, int]] = set()
    for x, y in lens:
        for pos in get_next_pos(x + y * width, width, width * height):
            if board[pos] != CellType.UNKNOWN:
                continue

            output.add((pos % width, pos // width))
    return list(output)


def pos_int_to_tuple(pos: int, width: int) -> Tuple[int, int]:
    """座標変換

    Parameters
    ----------
    pos
        座標
    width
        横幅

    Returns
    -------
        (x, y)
    """
    return pos % width, pos // width


def calc_reverse_lens(lens: List[Tuple[int, int]], from_point: Tuple[int, int], to_point: Tuple[int, int])\
        -> List[Tuple[int, int]]:
    """fromPointとtoPointを基準にした、lensの線対称図形を取得する

    Parameters
    ----------
    lens
        レンズ
    from_point
        レンズを反転する際の起点
    to_point
        レンズを反転する際の終点

    Returns
    -------
        反転後のレンズ
    """
    # 計算が面倒なので、from_pointとto_pointに対しての関係性から場合分けする
    offset = (to_point[0] - from_point[0], to_point[1] - from_point[1])
    output: List[Tuple[int, int]] = []
    if offset[0] == 0:
        # ブリッジが縦方向な場合
        for cell in lens:
            output.append((to_point[0] + cell[0] - from_point[0], to_point[1] - cell[1] + from_point[1]))
    elif offset[1] == 0:
        # ブリッジが横方向な場合
        for cell in lens:
            output.append((to_point[0] - cell[0] + from_point[0], to_point[1] + cell[1] - from_point[1]))
    else:
        # ブリッジが斜め方向な場合
        for cell in lens:
            output.append((to_point[0] + cell[1] - from_point[1], to_point[1] + cell[0] - from_point[0]))
    return output


def pattern_sync_bridge_lenses(board: List[CellType], problem: SunGlass) -> List[CellType]:
    """各ブリッジの双翼に生えているレンズの、塗りつぶし状態・上下左右の空白状態を同期
    同期不可能な場合は例外を投げる

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

    output = [x for x in board]
    for bridge in problem.bridge:
        # 左翼・右翼のレンズを取得する
        left_point = (bridge.cell[0].x, bridge.cell[0].y)
        right_point = (bridge.cell[-1].x, bridge.cell[-1].y)
        left_lens = [pos_int_to_tuple(x, problem.width) for x in
                     find_lens(board, problem.width, left_point[0] + left_point[1] * problem.width)]
        right_lens = [pos_int_to_tuple(x, problem.width) for x in
                      find_lens(board, problem.width, right_point[0] + right_point[1] * problem.width)]

        # 左翼・右翼のレンズについて、「レンズの周囲の空白マス」の一覧を取得する
        left_blank_cells = find_around_blank_cells(board, problem.width, problem.height, left_lens)
        right_blank_cells = find_around_blank_cells(board, problem.width, problem.height, right_lens)

        # レンズについて、塗りつぶし状態を同期する
        left_lens_reverse = calc_reverse_lens(left_lens, left_point, right_point)
        right_lens_reverse = calc_reverse_lens(right_lens, right_point, left_point)
        appending_lens_cells = (set(right_lens_reverse) - set(left_lens)) | (set(left_lens_reverse) - set(right_lens))
        for pos in appending_lens_cells:
            if 0 <= pos[0] < problem.width and 0 <= pos[1] < problem.height:
                if output[pos[0] + pos[1] * problem.width] == CellType.BLANK:
                    raise AlgorithmException('塗りつぶし状態を同期できません')
                else:
                    output[pos[0] + pos[1] * problem.width] = CellType.LENS

        # レンズの周囲の空白マスの状態を同期する
        left_blank_cells_reverse = calc_reverse_lens(left_blank_cells, left_point, right_point)
        right_blank_cells_reverse = calc_reverse_lens(right_blank_cells, right_point, left_point)
        appending_blank_cells = (set(right_blank_cells_reverse) - set(left_blank_cells)) |\
                                (set(left_blank_cells_reverse) - set(right_blank_cells))
        for pos in appending_blank_cells:
            if 0 <= pos[0] < problem.width and 0 <= pos[1] < problem.height:
                output[pos[0] + pos[1] * problem.width] = CellType.BLANK
    return output


def pattern_hint(board: List[CellType], problem: SunGlass) -> List[CellType]:
    """ヒント数字に従い、ちょうど塗りつぶせるなら塗り潰す

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

    output = [x for x in board]
    for hint in problem.hint:
        # レンズマスの数、空白マスの数、不明マスの位置を調べる
        lens_cells_count = 0
        blank_cells_count = 0
        unknown_cells: List[int] = []
        if hint.type == HintType.ROW:
            # 行ヒント
            pos = hint.index * problem.width
            for _ in range(0, problem.width):
                if board[pos] == CellType.LENS:
                    lens_cells_count += 1
                elif board[pos] == CellType.BLANK:
                    blank_cells_count += 1
                else:
                    unknown_cells.append(pos)
                pos += 1
        else:
            # 列ヒント
            pos = hint.index
            for _ in range(0, problem.height):
                if board[pos] == CellType.LENS:
                    lens_cells_count += 1
                elif board[pos] == CellType.BLANK:
                    blank_cells_count += 1
                else:
                    unknown_cells.append(pos)
                pos += problem.width

        # 不明マスの数＋レンズマスの数＝ヒントの数字ならば、
        # 不明マスが全てレンズマスであるはず
        if len(unknown_cells) + lens_cells_count == hint.value:
            for pos in unknown_cells:
                output[pos] = CellType.LENS

        # レンズマスの数＝ヒントの数字ならば、
        # 不明マスが全て空白マスであるはず
        if lens_cells_count == hint.value:
            for pos in unknown_cells:
                output[pos] = CellType.BLANK
    return output


def calc_symmetry_axis(left_point: Tuple[int, int], right_point: Tuple[int, int], width: int, height: int)\
        -> List[Tuple[int, int]]:
    """線対称の軸となるマスの座標を割り出す

    Parameters
    ----------
    left_point
        左側の座標
    right_point
        右側の座標
    width
        横幅
    height
        縦幅

    Returns
    -------
        線対称の軸となるマスの座標の一覧
    """

    # 左側・右側の相対関係によって場合分け。
    # ただし間隔が偶数な場合と奇数な場合とで処理が異なる
    output: List[Tuple[int, int]] = []
    if left_point[0] == right_point[0]:
        # 縦方向
        if (left_point[1] - right_point[1]) % 2 == 1:
            y1 = (abs(left_point[1] - right_point[1]) - 1) // 2 + min(left_point[1], right_point[1])
            y2 = y1 + 1
            for k in range(width):
                output.append((k, y1))
                output.append((k, y2))
        else:
            y = (abs(left_point[1] - right_point[1])) // 2 + min(left_point[1], right_point[1])
            for k in range(width):
                output.append((k, y))
        return output

    if left_point[1] == right_point[1]:
        # 横方向
        if (left_point[0] - right_point[0]) % 2 == 1:
            x1 = (abs(left_point[0] - right_point[0]) - 1) // 2 + min(left_point[0], right_point[0])
            x2 = x1 + 1
            for k in range(width):
                output.append((x1, k))
                output.append((x2, k))
        else:
            x = (abs(left_point[0] - right_point[0])) // 2 + min(left_point[0], right_point[0])
            for k in range(width):
                output.append((x, k))
        return output

    # 斜め方向
    if (left_point[0] - right_point[0]) * (left_point[1] - right_point[1]) > 0:
        slant_type = 'b'    # バックスラッシュ形
    else:
        slant_type = 's'    # スラッシュ形

    if (left_point[0] - right_point[0]) % 2 == 1:
        # 間隔が偶数なケース
        if slant_type == 'b':
            x = (abs(left_point[0] - right_point[0]) - 1) // 2 + min(left_point[0], right_point[0])
            y = (abs(left_point[1] - right_point[1]) - 1) // 2 + min(left_point[1], right_point[1]) + 1
            center_point = (x, y)
        else:
            x = (abs(left_point[0] - right_point[0]) - 1) // 2 + min(left_point[0], right_point[0])
            y = (abs(left_point[1] - right_point[1]) - 1) // 2 + min(left_point[1], right_point[1])
            center_point = (x, y)
    else:
        # 間隔が奇数なケース
        x = (abs(left_point[0] - right_point[0]) - 1) // 2 + min(left_point[0], right_point[0])
        y = (abs(left_point[1] - right_point[1]) - 1) // 2 + min(left_point[1], right_point[1])
        center_point = (x, y)

    if slant_type == 'b':
        output.append(center_point)
        temp = (center_point[0], center_point[1])
        while True:
            temp = (temp[0] + 1, temp[1] - 1)
            if 0 <= temp[0] < width and 0 <= temp[1] < height:
                output.append((temp[0], temp[1]))
            else:
                break
        temp = (center_point[0], center_point[1])
        while True:
            temp = (temp[0] - 1, temp[1] + 1)
            if 0 <= temp[0] < width and 0 <= temp[1] < height:
                output.append((temp[0], temp[1]))
            else:
                break
    else:
        output.append(center_point)
        temp = (center_point[0], center_point[1])
        while True:
            temp = (temp[0] + 1, temp[1] + 1)
            if 0 <= temp[0] < width and 0 <= temp[1] < height:
                output.append((temp[0], temp[1]))
            else:
                break
        temp = (center_point[0], center_point[1])
        while True:
            temp = (temp[0] - 1, temp[1] - 1)
            if 0 <= temp[0] < width and 0 <= temp[1] < height:
                output.append((temp[0], temp[1]))
            else:
                break

    return output


def pattern_can_not_reach(board: List[CellType], problem: SunGlass) -> List[CellType]:
    """どのブリッジからも塗りつぶせない位置のマスは空白マス

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

    """
    def print_lens(lens: List[Tuple[int, int]], star_mark=(-1, -1)):
        mark = '□■'
        arr = [0] * (problem.width * problem.height)
        for pos in lens:
            arr[pos[0] + pos[1] * problem.width] = 1
        for i in range(problem.width * problem.height):
            if i == star_mark[0] + star_mark[1] * problem.width:
                print(f'☆ ', end='')
            else:
                print(f'{mark[arr[i]]} ', end='')
            if i % problem.width == problem.width - 1:
                print('')
    """

    # それぞれのブリッジにおける、現在のレンズ情報を取得する
    lens_list: List[Tuple[Tuple[int, int], Tuple[int, int], Set[int], Set[int]]] = []
    all_lenses = set()
    for bridge in problem.bridge:
        # 左翼・右翼のレンズを取得する
        left_point = (bridge.cell[0].x, bridge.cell[0].y)
        right_point = (bridge.cell[-1].x, bridge.cell[-1].y)
        left_lens = set(find_lens(board, problem.width, left_point[0] + left_point[1] * problem.width))
        right_lens = set(find_lens(board, problem.width, right_point[0] + right_point[1] * problem.width))

        lens_list.append((left_point, right_point, left_lens, right_lens))
        all_lenses = all_lenses | left_lens | right_lens

    # それぞれのブリッジにおいて、伸ばせる最大のレンズの範囲を算出する
    all_max_lenses = set()
    output = [x for x in board]
    for left_point, right_point, left_lens, right_lens in lens_list:
        # 右翼・左翼について、「レンズを最大限伸ばした際の範囲」を取得する
        left_lens_max = [pos_int_to_tuple(x, problem.width) for x in
                         find_lens_max(board, problem.width, len(board), left_lens, all_lenses - left_lens)]
        right_lens_max = [pos_int_to_tuple(x, problem.width) for x in
                          find_lens_max(board, problem.width, len(board), right_lens, all_lenses - right_lens)]

        """
        print(f'[[left_lens_max(left_point={left_point})]]')
        print_lens(list(left_lens_max), left_point)
        print(f'[[right_lens_max(right_point={right_point})]]')
        print_lens(list(right_lens_max), right_point)
        """

        # レンズは左右対称になるので、左右対称にできない部分は「レンズを最大限伸ばした際の範囲」から削れる
        # また、ブリッジにおける線対称の軸にあたる部分は、当然塗ることができない
        left_lens_max_reverse = calc_reverse_lens(left_lens_max, left_point, right_point)
        right_lens_max_reverse = calc_reverse_lens(right_lens_max, right_point, left_point)
        symmetry_axis = calc_symmetry_axis(left_point, right_point, problem.width, problem.height)
        left_lens_max = (set(left_lens_max) & set(right_lens_max_reverse)) - set(symmetry_axis)
        right_lens_max = (set(right_lens_max) & set(left_lens_max_reverse)) - set(symmetry_axis)

        """
        if (1, 6) not in left_lens_max:
            print('[board]')
            show_board_data(board, problem)
            print(f'symmetry_axis')
            print_lens(symmetry_axis)
            print(f'left_lens_max_reverse')
            print_lens(left_lens_max_reverse, right_point)
            print(f'right_lens_max_reverse')
            print_lens(right_lens_max_reverse, left_point)
            print(f'[left_lens_max(left_point={left_point})]')
            print_lens(list(left_lens_max), left_point)
            print(f'[right_lens_max(right_point={right_point})]')
            print_lens(list(right_lens_max), right_point)
        """

        # 「レンズを最大限伸ばした際の範囲」をmergeする
        all_max_lenses = all_max_lenses | left_lens_max | right_lens_max

    # どのレンズからも被覆できない部分は、当然空白マスになる
    for y in range(problem.height):
        for x in range(problem.width):
            if board[x + y * problem.width] == CellType.UNKNOWN and (x, y) not in all_max_lenses:
                output[x + y * problem.width] = CellType.BLANK
    return output


def solve_by_tactics(board: List[CellType], problem: SunGlass, show_log_flg=True) -> List[CellType]:
    """各種定石を使用して盤面を埋める

    Parameters
    ----------
    board
        盤面
    problem
        問題
    show_log_flg
        途中経過を表示するならTrue

    Returns
    -------
        埋めた後の盤面
    """
    output_board = board.copy()
    while True:
        # レンズ同士が接触しないように空白を設置
        next_board = pattern_do_not_join_lenses(output_board, problem)
        if not is_equal(output_board, next_board):
            output_board = next_board
            if show_log_flg:
                print('・レンズ同士が接触しないように空白を設置')
                show_board_data(output_board, problem)
            continue

        # 各ブリッジの双翼に生えているレンズの、塗りつぶし状態・上下左右の空白状態を同期
        next_board = pattern_sync_bridge_lenses(output_board, problem)
        if not is_equal(output_board, next_board):
            output_board = next_board
            if show_log_flg:
                print('・塗りつぶし状態・上下左右の空白状態を同期')
                show_board_data(output_board, problem)
            continue

        # ヒント数字に従い、ちょうど塗りつぶせるなら塗り潰す
        next_board = pattern_hint(output_board, problem)
        if not is_equal(output_board, next_board):
            output_board = next_board
            if show_log_flg:
                print('・ヒント数字に従い塗りつぶせるなら塗り潰す')
                show_board_data(output_board, problem)
            continue

        # どのブリッジからも塗りつぶせない位置のマスは空白マス
        next_board = pattern_can_not_reach(output_board, problem)
        if not is_equal(output_board, next_board):
            output_board = next_board
            if show_log_flg:
                print('・どこからも塗り潰せなければそこは空白マス')
                show_board_data(output_board, problem)
            continue
        break

    return output_board


def is_invalid_board(board: List[CellType], problem: SunGlass) -> bool:
    """矛盾が存在する盤面の場合はTrue

    Parameters
    ----------
    board
        盤面
    problem
        問題

    Returns
    -------
        矛盾があればTrue
    """

    # ヒント数字の条件をどうしても満たせない場合はアウト
    for hint in problem.hint:
        if hint.type == HintType.ROW:
            # 行についての確認
            lens_count = 0
            unknown_count = 0
            for k in range(problem.width):
                cell = board[k + hint.index * problem.width]
                if cell == CellType.LENS:
                    lens_count += 1
                elif cell == CellType.UNKNOWN:
                    unknown_count += 1
            if lens_count + unknown_count < hint.value:
                return True
            if lens_count > hint.value:
                return True
        else:
            # 列についての確認
            lens_count = 0
            unknown_count = 0
            for k in range(problem.height):
                cell = board[hint.index + k * problem.width]
                if cell == CellType.LENS:
                    lens_count += 1
                elif cell == CellType.UNKNOWN:
                    unknown_count += 1
            if lens_count + unknown_count < hint.value:
                return True
            if lens_count > hint.value:
                return True

    # どのブリッジにも属せない、孤立したレンズが存在する場合はアウト
    bridge_cells: Set[int] = set()
    for bridge in problem.bridge:
        bridge_cells.add(bridge.cell[0].x + bridge.cell[0].y * problem.width)
        bridge_cells.add(bridge.cell[-1].x + bridge.cell[-1].y * problem.width)

    for i in range(len(board)):
        # どのブリッジにも属していない、孤立レンズを検索する
        if board[i] != CellType.LENS:
            continue
        lens = set(find_lens(board, problem.width, i))
        if len(lens & bridge_cells) != 0:
            continue

        # 孤立レンズについて、その周囲に不明マスが存在するかを確認する
        lens_xy = [pos_int_to_tuple(k, problem.width) for k in lens]
        if len(find_around_unknown_cells(board, problem.width, problem.height, lens_xy)) == 0:
            # 孤立レンズの周囲に不明マスがない＝ブリッジと接続できないのでアウト
            return True

    return False


def is_invalid_board_wrapper(input_board: List[CellType], problem: SunGlass) -> bool:
    try:
        temp_board = solve_by_tactics(input_board, problem, False)
        return is_invalid_board(temp_board, problem)
    except AlgorithmException:
        return True


def is_invalid_board_wrapper_2(input_board: List[CellType], problem: SunGlass) -> bool:
    try:
        temp_board = solve_by_1st_absurdum(input_board, problem, False)
        return is_invalid_board(temp_board, problem)
    except AlgorithmException:
        return True


def solve_by_1st_absurdum(board: List[CellType], problem: SunGlass, show_log_flg=True) -> List[CellType]:
    """1段階背理法

    Parameters
    ----------
    board
        盤面
    problem
        問題
    show_log_flg
        途中経過を表示するならTrue

    Returns
    -------
        埋めたあとの盤面
    """

    output_board = board.copy()
    while True:
        # 定石で盤面を埋める
        board2 = solve_by_tactics(output_board, problem, show_log_flg)
        if not is_equal(output_board, board2):
            output_board = board2
            continue

        if show_log_flg:
            print('(背理法開始)')
        flg = False
        for i in range(len(output_board)):
            if output_board[i] != CellType.UNKNOWN:
                continue
            # print(f'(座標{pos_int_to_tuple(i, problem.width)}に背理法)')

            # レンズと仮定して、矛盾すればそれはレンズではない
            output_board[i] = CellType.LENS
            if is_invalid_board_wrapper(output_board, problem):
                output_board[i] = CellType.BLANK
                if show_log_flg:
                    print(f'・背理法　座標{pos_int_to_tuple(i, problem.width)}を空白に')
                    show_board_data(output_board, problem)
                flg = True
                break

            # 空白と仮定して、矛盾すればそれは空白ではない
            output_board[i] = CellType.BLANK
            if is_invalid_board_wrapper(output_board, problem):
                output_board[i] = CellType.LENS
                if show_log_flg:
                    print(f'・背理法　座標{pos_int_to_tuple(i, problem.width)}をレンズに')
                    show_board_data(output_board, problem)
                flg = True
                break

            output_board[i] = CellType.UNKNOWN
        if not flg:
            if show_log_flg:
                print('(背理法終了)')
            break

    return output_board


def solve_by_2nd_absurdum(board: List[CellType], problem: SunGlass, show_log_flg=True) -> List[CellType]:
    """2段階背理法

    Parameters
    ----------
    board
        盤面
    problem
        問題
    show_log_flg
        途中経過を表示するならTrue

    Returns
    -------
        埋めたあとの盤面
    """

    output_board = board.copy()
    while True:
        # 1段階背理法で盤面を埋める
        board2 = solve_by_1st_absurdum(output_board, problem, show_log_flg)
        if not is_equal(output_board, board2):
            output_board = board2
            continue

        if show_log_flg:
            print('(2段階背理法開始)')
        flg = False
        for i in range(len(output_board)):
            if output_board[i] != CellType.UNKNOWN:
                continue
            # print(f'(座標{pos_int_to_tuple(i, problem.width)}に背理法)')

            # レンズと仮定して、矛盾すればそれはレンズではない
            output_board[i] = CellType.LENS
            if is_invalid_board_wrapper_2(output_board, problem):
                output_board[i] = CellType.BLANK
                if show_log_flg:
                    print(f'・背理法　座標{pos_int_to_tuple(i, problem.width)}を空白に')
                    show_board_data(output_board, problem)
                flg = True
                break

            # 空白と仮定して、矛盾すればそれは空白ではない
            output_board[i] = CellType.BLANK
            if is_invalid_board_wrapper_2(output_board, problem):
                output_board[i] = CellType.LENS
                if show_log_flg:
                    print(f'・背理法　座標{pos_int_to_tuple(i, problem.width)}をレンズに')
                    show_board_data(output_board, problem)
                flg = True
                break

            output_board[i] = CellType.UNKNOWN
        if not flg:
            if show_log_flg:
                print('(2段階背理法終了)')
            break

    return output_board


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
    print('【問題】')
    show_board_data(board, problem)

    # 解析
    board2 = solve_by_2nd_absurdum(board, problem)

    print('【結果】')
    show_board_data(board2, problem)


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
