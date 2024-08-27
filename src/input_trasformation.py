import torch


class InputTransformer:
    """
    計算安定性のために，目的変数を正規化する．
    その他，実験の際に使用する変換処理を提供．
    """

    def __init__(self, search_space, lower_bound=-1, upper_bound=1) -> None:
        self.x_min = search_space[0]
        self.x_max = search_space[1]
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def normalize(self, x: torch.tensor) -> torch.tensor:
        if self.x_max is None:
            self.x_max = x.max()
        if self.x_min is None:
            self.x_min = x.min()
        return (x - self.x_min) / (self.x_max - self.x_min) * (
            self.upper_bound - self.lower_bound
        ) + self.lower_bound

    def denormalize(self, x: torch.tensor) -> torch.tensor:
        return (x - self.lower_bound) / (self.upper_bound - self.lower_bound) * (
            self.x_max - self.x_min
        ) + self.x_min

    def discretize(self, x: torch.tensor) -> torch.tensor:
        return x.round()

    def clipping(self, x: torch.tensor) -> torch.tensor:
        """
        各次元ごとに異なる範囲でのクリッピングを可能にする
        """
        return torch.max(torch.min(x, self.x_max), self.x_min)
