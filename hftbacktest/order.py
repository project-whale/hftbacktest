import numpy as np
from numba import float64, int64, int8, boolean
from numba.experimental import jitclass
from numba.typed import Dict, List
from numba.types import DictType, ListType, Tuple


BUY = 1
SELL = -1

NONE = 0
NEW = 1
EXPIRED = 2
FILLED = 3
CANCELED = 4
PARTIALLY_FILLED = 5
MODIFY = 6
REJECTED = 7

GTC = 0  # Good 'till cancel
GTX = 1  # Post only
FOK = 2  # Fill or kill
IOC = 3  # Immediate or cancel

LIMIT = 0
MARKET = 1


@jitclass
class Order:
    qty: float64
    leaves_qty: float64
    price_tick: int64
    tick_size: float64
    side: int8
    time_in_force: int8
    exch_timestamp: int64
    status: int8
    local_timestamp: int64
    req: int8
    exec_price_tick: int64
    exec_qty: float64
    order_id: int64
    q: float64[:]
    maker: boolean
    order_type: int8

    def __init__(
            self,
            order_id,
            price_tick,
            tick_size,
            qty,
            side,
            time_in_force,
            order_type
    ):
        self.qty = qty
        self.leaves_qty = qty
        self.price_tick = price_tick
        self.tick_size = tick_size
        self.side = side
        self.time_in_force = time_in_force
        self.exch_timestamp = 0
        self.status = NONE
        self.local_timestamp = 0
        self.req = NONE
        self.exec_price_tick = 0
        self.exec_qty = 0.0
        self.order_id = order_id
        self.q = np.zeros(2, float64)
        self.maker = False
        self.order_type = order_type

    @property
    def limit(self):
        # compatibility <= 1.3
        return self.maker

    @property
    def price(self):
        return self.price_tick * self.tick_size

    @property
    def exec_price(self):
        return self.exec_price_tick * self.tick_size

    @property
    def cancellable(self):
        return self.status == NEW and self.req == NONE

    def copy(self):
        """
        Return copy of current instance of Order class with current attributes
        """
        order = Order(
            self.order_id,
            self.price_tick,
            self.tick_size,
            self.qty,
            self.side,
            self.time_in_force,
            self.order_type
        )
        order.leaves_qty = self.leaves_qty
        order.exch_timestamp = self.exch_timestamp
        order.status = self.status
        order.local_timestamp = self.local_timestamp
        order.req = self.req
        order.exec_price_tick = self.exec_price_tick
        order.exec_qty = self.exec_qty
        order.order_id = self.order_id
        order.q[:] = self.q[:]
        order.maker = self.maker
        return order


order_ty = Order.class_type.instance_type
order_ladder_ty = DictType(int64, order_ty)
order_tup_ty = Tuple((order_ty, int64))


@jitclass
class OrderBus:
    """
    OrderBus class represents a bus for managing orders in a high-frequency trading backtesting system.

    Attributes:
        order_list (ListType): A list of order tuples.
        orders (DictType): A dictionary of order IDs and their counts.
        frontmost_timestamp (int64): The frontmost timestamp of the order list.

    Methods:
        __init__: Initializes the OrderBus object.
        append: Appends an order to the order list.
        get: Retrieves the receive timestamp of an order by its order ID.
        reset: Resets the order list and order dictionary.
        __getitem__: Retrieves an order tuple by its index.
        __len__: Returns the length of the order list.
        delitem: Deletes an order tuple by its index.
        __contains__: Checks if an order ID exists in the order dictionary.
    """
    order_list: ListType(order_tup_ty)
    orders: DictType(int64, int64)
    frontmost_timestamp: int64

    def __init__(self):
        """
        Initializes the OrderBus object.

        Parameters:
            None

        Returns:
            None
        """
        self.order_list = List.empty_list(order_tup_ty)
        self.orders = Dict.empty(int64, int64)
        self.frontmost_timestamp = 0

    def append(self, order, timestamp):
        """
        Appends an order to the order list.

        Parameters:
            order (Order): The order to be appended.
            timestamp (int): The timestamp of the order.

        Returns:
            None
        """
        timestamp = int(timestamp)

        if len(self.order_list) > 0:
            _, latest_timestamp = self.order_list[-1]
            if timestamp < latest_timestamp:
                timestamp = latest_timestamp

        self.order_list.append((order, timestamp))

        if order.order_id in self.orders:
            self.orders[order.order_id] += 1
        else:
            self.orders[order.order_id] = 1

        if self.frontmost_timestamp <= 0:
            self.frontmost_timestamp = timestamp
        else:
            self.frontmost_timestamp = min(self.frontmost_timestamp, timestamp)

    def get(self, order_id):
        """
        Retrieves the receive timestamp of an order by its order ID.

        Parameters:
            order_id (int): The order ID.

        Returns:
            int: The receive timestamp of the order.

        Raises:
            KeyError: If the order ID does not exist in the order list.
        """
        for order, recv_timestamp in self.order_list:
            if order.order_id == order_id:
                return recv_timestamp
        raise KeyError

    def reset(self):
        """
        Resets the order list and order dictionary.

        Parameters:
            None

        Returns:
            None
        """
        self.order_list.clear()
        self.orders.clear()
        self.frontmost_timestamp = 0

    def __getitem__(self, key):
        """
        Retrieves an order tuple by its index.

        Parameters:
            key (int): The index of the order tuple.

        Returns:
            Tuple[Order, int]: The order tuple.

        Raises:
            IndexError: If the index is out of range.
        """
        return self.order_list[key]

    def __len__(self):
        """
        Returns the length of the order list.

        Parameters:
            None

        Returns:
            int: The length of the order list.
        """
        return len(self.order_list)

    def delitem(self, key):
        """
        Deletes an order tuple by its index.

        Parameters:
            key (int): The index of the order tuple.

        Returns:
            None

        Raises:
            IndexError: If the index is out of range.
        """
        order, _ = self.order_list[key]
        del self.order_list[key]
        self.orders[order.order_id] -= 1
        if self.orders[order.order_id] <= 0:
            del self.orders[order.order_id]

    def __contains__(self, key):
        """
        Checks if an order ID exists in the order dictionary.

        Parameters:
            key (int): The order ID.

        Returns:
            bool: True if the order ID exists, False otherwise.
        """
        return key in self.orders
