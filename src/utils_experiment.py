import re
import torch


def extract_info_from_filename(filename):
    # Extract date and objective function from the filename
    match = re.match(r'.*(\d{4}-\d{2}-\d{2})_(\w+)_v\d+\.py', filename)
    if match:
        date = match.group(1)
        objective_function = match.group(2)
        return date, objective_function
    return None, None


def generate_integer_samples(bounds, n, device=torch.device("cpu")):
    """
    整数をランダムにサンプリングして、n 行 m 列の torch.Tensor を生成します。
    重複のない n サンプルが得られるまでサンプリングを繰り返します。
    
    Parameters:
    - bounds: list of list, 変数の下限と上限のリスト
    - n: int, サンプル数
    - device: torch.device, テンソルを配置するデバイス
    
    Returns:
    - torch.Tensor, n 行 m 列のテンソル
    """
    lower_bounds = torch.tensor(bounds[0], device=device, dtype=torch.int)
    upper_bounds = torch.tensor(bounds[1], device=device, dtype=torch.int)

    m = lower_bounds.shape[0]
    samples = set()

    while len(samples) < n:
        new_samples = []
        for _ in range(n):
            sample = []
            for i in range(m):
                sample.append(torch.randint(low=lower_bounds[i].item(), high=upper_bounds[i].item() + 1, size=(1,), device=device).item())
            new_samples.append(tuple(sample))
        
        for sample in new_samples:
            samples.add(sample)

        if len(samples) >= n:
            break

    unique_samples = torch.tensor(list(samples)[:n], device=device)
    return unique_samples


def negate_function(func):
    """ 
    目的関数の符号を反転させる
    """
    def negated_func(X):
        return -func(X)
    return negated_func



# if __name__ == "__main__":
#     # テスト
#     bounds = [[0, 0, 0, 0, 0, 0], [4, 5, 6, 7, 8, 9]]
#     n = 10
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # samples = generate_integer_samples(bounds, n, device)
#     # print(samples)
#     samples = generate_integer_samples(bounds, n)
#     print(samples)