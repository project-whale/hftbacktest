from numba import int64


class LinearAsset:
    """
    Linear asset: the common type of asset.

    Args:
        contract_size (int): Contract size of the asset.
    """

    contract_size: int64

    def __init__(self, contract_size=1):
        self.contract_size = contract_size

    def amount(self, exec_price, qty):
        return self.contract_size * exec_price * qty

    def equity(self, price, balance, position, fee):
        return balance + self.contract_size * position * price - fee


class InverseAsset:
    """
    Inverse asset: the contract's notional value is denominated in the quote currency.

    Args:
        contract_size (int): Contract size of the asset.

    Attributes:
        contract_size (int): Contract size of the asset.

    Methods:
        __init__(self, contract_size=1): Initializes an instance of InverseAsset.
        amount(self, exec_price, qty): Calculates the amount of the asset based on the execution price and quantity.
        equity(self, price, balance, position, fee): Calculates the equity of the asset based on the price, balance, position, and fee.
    """

    contract_size: int64

    def __init__(self, contract_size=1):
        self.contract_size = contract_size

    def amount(self, exec_price, qty):
        return self.contract_size * qty / exec_price

    def equity(self, price, balance, position, fee):
        return -balance - self.contract_size * position / price - fee
