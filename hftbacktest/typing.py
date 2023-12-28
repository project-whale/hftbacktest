from typing import Union, Callable, NewType, List

from numba.experimental.jitclass.base import JitClassType
from numpy.typing import NDArray
from pandas import DataFrame


Data = Union[str, NDArray, DataFrame]
DataCollection = Union[Data, List[Data]]

HftBacktestType = NewType('HftBacktest', JitClassType)
Reader = NewType('Reader', JitClassType)
OrderBus = NewType('OrderBus', JitClassType)
MarketDepth = NewType('MarketDepth', JitClassType)
State = NewType('State', JitClassType)
OrderLatencyModel = NewType('OrderLatencyModel', JitClassType)
AssetType = NewType('AssetType', JitClassType)
QueueModel = NewType('QueueModel', JitClassType)

ExchangeModelInitiator = Callable[
    [
        Reader,
        OrderBus,
        OrderBus,
        MarketDepth,
        State,
        OrderLatencyModel,
        QueueModel
    ],
    JitClassType
]

"""
This module defines type aliases used in the hftbacktest package.

- Data: Represents the data type that can be used for backtesting.
- DataCollection: Represents a collection of data.
- HftBacktestType: Represents the type of a high-frequency trading backtest.
- Reader: Represents the type of a reader.
- OrderBus: Represents the type of an order bus.
- MarketDepth: Represents the type of market depth data.
- State: Represents the type of a state.
- OrderLatencyModel: Represents the type of an order latency model.
- AssetType: Represents the type of an asset.
- QueueModel: Represents the type of a queue model.
- ExchangeModelInitiator: Represents a callable that initializes an exchange model.

Please refer to the documentation for more information on each type.
"""
