import numpy as np
from numba import typeof, float64, int64
from numba.typed import Dict

from ..marketdepth import MarketDepth
from ..order import order_ladder_ty, order_ty, OrderBus


class Proc:
    """
    The Proc class represents a processor that handles data and orders in a high-frequency trading backtesting system.

    Attributes:
        reader: The data reader object.
        next_data: The next data to be processed.
        data: The current data being processed.
        row_num: The current row number in the data.
        next_row_num: The next row number in the data.
        orders: A dictionary that stores the orders.
        orders_to: The order bus for sending orders.
        orders_from: The order bus for receiving orders.
        depth: The depth object that stores the bid and ask depth.
        state: The state object that stores the trading state.
        order_latency: The order latency object that measures the latency of orders.

    Methods:
        _proc_init: Initializes the processor with the necessary parameters.
        _proc_reset: Resets the processor to the initial state.
        next_timestamp: Returns the next valid timestamp.
        _next_data_timestamp_column: Finds the next valid timestamp in a specific column.
        process: Processes the data and orders.
    """
    def __init__(self):
        pass

    def _proc_init(self, reader, orders_to, orders_from, depth, state, order_latency):
        """
        Initializes the processor with the necessary parameters.

        Args:
            reader: The data reader object.
            orders_to: The order bus for sending orders.
            orders_from: The order bus for receiving orders.
            depth: The depth object that stores the bid and ask depth.
            state: The state object that stores the trading state.
            order_latency: The order latency object that measures the latency of orders.
        """
        self.reader = reader
        self.next_data = reader.next()
        self.data = np.empty((0, self.next_data.shape[1]), self.next_data.dtype)
        self.row_num = 0
        self.next_row_num = 0

        self.orders = Dict.empty(int64, order_ty)

        self.orders_to = orders_to
        self.orders_from = orders_from

        self.depth = depth
        self.state = state
        self.order_latency = order_latency

    def _proc_reset(
            self,
            start_position,
            start_balance,
            start_fee,
            maker_fee,
            taker_fee,
            tick_size,
            lot_size,
            snapshot
    ):
        """
        Resets the processor to the initial state.

        Args:
            start_position: The starting position.
            start_balance: The starting balance.
            start_fee: The starting fee.
            maker_fee: The maker fee.
            taker_fee: The taker fee.
            tick_size: The tick size.
            lot_size: The lot size.
            snapshot: The snapshot of the depth.
        """
        self.next_data = self.reader.next()
        self.data = np.empty((0, self.next_data.shape[1]), self.next_data.dtype)
        self.row_num = 0
        self.next_row_num = 0

        self.orders.clear()

        self.orders_to.reset()
        self.orders_from.reset()

        self.depth.clear_depth(0, 0)

        if tick_size is not None:
            self.depth.tick_size = tick_size

        if lot_size is not None:
            self.depth.lot_size = lot_size

        if snapshot is not None:
            self.depth.apply_snapshot(snapshot)

        self.state.reset(start_position, start_balance, start_fee, maker_fee, taker_fee)
        self.order_latency.reset()

    def next_timestamp(self):
        """
        Returns the next valid timestamp.

        Returns:
            The next valid timestamp.
        """
        next_data_timestamp = self._next_data_timestamp()
        next_recv_order_timestamp = self.orders_from.frontmost_timestamp

        # zero and negative timestamp are invalid timestamp.
        # if two timestamps are valid, choose the earlier one.
        # otherwise, choose the valid one.
        if (0 < next_recv_order_timestamp < next_data_timestamp) \
                or (next_data_timestamp <= 0 < next_recv_order_timestamp):
            return next_recv_order_timestamp
        else:
            return next_data_timestamp

    def _next_data_timestamp_column(self, column):
        """
        Finds the next valid timestamp in a specific column.

        Args:
            column: The column index.

        Returns:
            The next valid timestamp in the specified column.
        """
        # Finds the next valid timestamp
        while True:
            if self.next_row_num < len(self.next_data):
                timestamp = self.next_data[self.next_row_num, column]
                if timestamp > 0:
                    return timestamp
            else:
                # If there is no next_data, return an invalid timestamp.
                if len(self.next_data) == 0:
                    return -2

                # Release the current next_data and load the next next_data.
                self.reader.release(self.next_data)
                self.next_data = self.reader.next()
                self.next_row_num = 0
                if len(self.next_data) == 0:
                    return -2

                timestamp = self.next_data[self.next_row_num, column]
                if timestamp > 0:
                    return timestamp
            self.next_row_num += 1

    def process(self, wait_resp):
        """
        Processes the data and orders.

        Args:
            wait_resp: Whether to wait for a response from the order bus.

        Returns:
            The next timestamp.
        """
        next_data_timestamp = self._next_data_timestamp()
        next_recv_order_timestamp = self.orders_from.frontmost_timestamp

        # zero and negative timestamp are invalid timestamp.
        # if two timestamps are valid, choose the earlier one.
        # otherwise, choose the valid one.
        if (0 < next_recv_order_timestamp < next_data_timestamp) \
                or (next_data_timestamp <= 0 < next_recv_order_timestamp):
            # Processes the order part.
            next_timestamp = 0
            next_frontmost_timestamp = 0
            while self.orders_from.__len__() > 0:
                order, recv_timestamp = self.orders_from[0]
                if recv_timestamp <= self.orders_from.frontmost_timestamp:
                    self.orders_from.delitem(0)

                    next_timestamp = self._process_recv_order(
                        order,
                        recv_timestamp,
                        wait_resp,
                        next_timestamp
                    )
                else:
                    # Since we enforce the order of received timestamps to be sequential in OrderBus's append method,
                    # the next received timestamp is the next frontmost timestamp.
                    next_frontmost_timestamp = recv_timestamp
                    break
            self.orders_from.frontmost_timestamp = next_frontmost_timestamp
            return next_timestamp
        else:
            # Processes the data part.
            # Moves to the next row.
            self.row_num = self.next_row_num
            self.data = self.next_data

            row = self.data[self.row_num]

            self.next_row_num += 1

            return self._process_data(row)

    @property
    def tick_size(self):
        """
        The tick size property.

        Returns:
            The tick size.
        """
        return self.depth.tick_size

    @property
    def lot_size(self):
        """
        The lot size property.

        Returns:
            The lot size.
        """
        return self.depth.lot_size

    @property
    def bid_depth(self):
        """
        The bid depth property.

        Returns:
            The bid depth.
        """
        return self.depth.bid_depth

    @property
    def ask_depth(self):
        """
        The ask depth property.

        Returns:
            The ask depth.
        """
        return self.depth.ask_depth


def proc_spec(reader, state, order_latency):
    return [
        ('reader', typeof(reader)),
        ('data', float64[:, :]),
        ('next_data', float64[:, :]),
        ('row_num', int64),
        ('next_row_num', int64),
        ('data_num', int64),
        ('next_data_num', int64),

        ('orders', order_ladder_ty),

        ('orders_to', OrderBus.class_type.instance_type),
        ('orders_from', OrderBus.class_type.instance_type),

        ('depth', MarketDepth.class_type.instance_type),
        ('state', typeof(state)),
        ('order_latency', typeof(order_latency)),
    ]
