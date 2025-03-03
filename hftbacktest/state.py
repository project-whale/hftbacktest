from numba import typeof, float64, int64
from numba.experimental import jitclass


class State_:
    def __init__(
            self,
            start_position,
            start_balance,
            start_fee,
            maker_fee,
            taker_fee,
            asset_type
    ):
        """
        Initialize the State_ object.

        Args:
            start_position (float): The starting position.
            start_balance (float): The starting balance.
            start_fee (float): The starting fee.
            maker_fee (float): The fee for maker orders.
            taker_fee (float): The fee for taker orders.
            asset_type (object): The asset type object.

        """
        self.position = start_position
        self.balance = start_balance
        self.fee = start_fee
        self.trade_num = 0
        self.trade_qty = 0
        self.trade_amount = 0
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.asset_type = asset_type

    def apply_fill(self, order):
        """
        Apply a fill to the state.

        Args:
            order (object): The order object.

        """
        fee = self.maker_fee if order.maker else self.taker_fee
        amount = self.asset_type.amount(order.exec_price, order.exec_qty)
        self.position += (order.exec_qty * order.side)
        self.balance -= (amount * order.side)
        self.fee += (amount * fee)
        self.trade_num += 1
        self.trade_qty += order.exec_qty
        self.trade_amount += amount

    def equity(self, mid):
        """
        Calculate the equity of the state.

        Args:
            mid (float): The mid price.

        Returns:
            float: The equity.

        """
        return self.asset_type.equity(mid, self.balance, self.position, self.fee)

    def reset(self, start_position, start_balance, start_fee, maker_fee, taker_fee):
        """
        Reset the state to the initial values.

        Args:
            start_position (float): The starting position.
            start_balance (float): The starting balance.
            start_fee (float): The starting fee.
            maker_fee (float): The fee for maker orders.
            taker_fee (float): The fee for taker orders.

        """
        self.position = start_position
        self.balance = start_balance
        self.fee = start_fee
        self.trade_num = 0
        self.trade_qty = 0
        self.trade_amount = 0
        if maker_fee is not None:
            self.maker_fee = maker_fee
        if taker_fee is not None:
            self.taker_fee = taker_fee


def State(
        start_position,
        start_balance,
        start_fee,
        maker_fee,
        taker_fee,
        asset_type
):
    jitted = jitclass(spec=[
        ('position', float64),
        ('balance', float64),
        ('fee', float64),
        ('trade_num', int64),
        ('trade_qty', float64),
        ('trade_amount', float64),
        ('maker_fee', float64),
        ('taker_fee', float64),
        ('asset_type', typeof(asset_type))
    ])(State_)
    return jitted(
        start_position,
        start_balance,
        start_fee,
        maker_fee,
        taker_fee,
        asset_type
    )
