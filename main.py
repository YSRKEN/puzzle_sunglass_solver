import json


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
        data = json.load(f)
        print(data)


if __name__ == '__main__':
    solve_from_path('sample_problem.json')
