==========
Order Fill
==========

Exchange Models
===============

HftBacktest is a market-data replay-based backtesting tool, which means your order cannot make any changes to the
simulated market, no market impact is considered. Therefore, one of the most important assumptions is that your order is
small enough not to make any market impact. In the end, you must test it in a live market with real market participants
and adjust your backtesting based on the discrepancies between the backtesting results and the live outcomes.

Hftbacktest offers two types of exchange simulation. `NoPartialFillExchange`_ is the default exchange simulation where
no partial fills occur. `PartialFillExchange`_ is the extended exchange simulation that accounts for partial fills in
specific cases. Since the market-data replay-based backtesting cannot alter the market, some partial fill cases may
still be unrealistic, such as taking market liquidity. This is because even if your order takes market liquidity, the
replayed market data's market depth and trades cannot change. It is essential to understand the underlying assumptions
in each backtesting simulation.

NoPartialFillExchange
---------------------

Conditions for Full Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Buy order in the order book

* Your order price >= the best ask price
* Your order price > sell trade price
* Your order is at the front of the queue && your order price == sell trade price

Sell order in the order book

* Your order price <= the best bid price
* Your order price < buy trade price
* Your order is at the front of the queue && your order price == buy trade price

Liquidity-Taking Order
~~~~~~~~~~~~~~~~~~~~~~

    Regardless of the quantity at the best, liquidity-taking orders will be fully executed at the best. Be aware that
    this may cause unrealistic fill simulations if you attempt to execute a large quantity.

Usage
~~~~~

..  code-block:: python

    from hftbacktest import NoPartialFillExchange

    hbt = HftBacktest(
        data,
        tick_size=0.01,
        lot_size=0.001,
        maker_fee=-0.00005,
        taker_fee=0.0007,
        exchange_model=NoPartialFillExchange,  # Default
        asset_type=Linear
    )

PartialFillExchange
-------------------

Conditions for Full Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Buy order in the order book

* Your order price >= the best ask price
* Your order price > sell trade price

Sell order in the order book

* Your order price <= the best bid price
* Your order price < buy trade price

Conditions for Partial Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Buy order in the order book

* Filled by (remaining) sell trade quantity: your order is at the front of the queue && your order price == sell
  trade price

Sell order in the order book

* Filled by (remaining) buy trade quantity: your order is at the front of the queue && your order price == buy trade
  price

Liquidity-Taking Order
~~~~~~~~~~~~~~~~~~~~~~

    Liquidity-taking orders will be executed based on the quantity of the order book, even though the best price and
    quantity do not change due to your execution. Be aware that this may cause unrealistic fill simulations if you
    attempt to execute a large quantity.

Usage
~~~~~

..  code-block:: python

    from hftbacktest import PartialFillExchange

    hbt = HftBacktest(
        data,
        tick_size=0.01,
        lot_size=0.001,
        maker_fee=-0.00005,
        taker_fee=0.0007,
        exchange_model=PartialFillExchange,
        asset_type=Linear
    )

Queue Models
============

Knowing your order's queue position is important to achieve accurate order fill simulation in backtesting depending on
the liquidity of an order book and trading activities.
If an exchange doesn't provide Market-By-Order, you have to guess it by modeling.
HftBacktest currently only supports Market-By-Price that is most crypto exchanges provide and it provides the following
queue position models for order fill simulation.

Please refer to the details at :doc:`Queue Models <reference/queue_models>`.

.. image:: images/liquidity-and-trade-activities.png

RiskAverseQueueModel
--------------------

This model is the most conservative model in terms of the chance of fill in the queue.
The decrease in quantity by cancellation or modification in the order book happens only at the tail of the queue so your
order queue position doesn't change.
The order queue position will be advanced only if a trade happens at the price.

..  code-block:: python

    from hftbacktest import RiskAverseQueueModel

    hbt = HftBacktest(
        data,
        tick_size=0.01,
        lot_size=0.001,
        maker_fee=-0.00005,
        taker_fee=0.0007,
        order_latency=IntpOrderLatency(latency_data),
        queue_model=RiskAverseQueueModel()  # Default
        asset_type=Linear
    )



ProbQueueModel
--------------
Based on a probability model according to your current queue position, the decrease in quantity happens at both before
and after the queue position.
So your queue position is also advanced according to the probability.
This model is implemented as described in

* https://quant.stackexchange.com/questions/3782/how-do-we-estimate-position-of-our-order-in-order-book
* https://rigtorp.se/2013/06/08/estimating-order-queue-position.html

By default, three variations are provided. These three models have different probability profiles.

.. image:: images/probqueuemodel.png

The function f = log(1 + x) exhibits a different probability profile depending on the total quantity at the price level,
unlike power functions.

.. image:: images/probqueuemodel_log.png

.. image:: images/probqueuemodel2.png
.. image:: images/probqueuemodel3.png

When you set the function f, it should be as follows.

* The probability at 0 should be 0 because if the order is at the head of the queue, all decreases should happen after
the order.
* The probability at 1 should be 1 because if the order is at the tail of the queue, all decreases should happen before
the order.

You can see the comparison of the models :doc:`here <tutorials/Probability Queue Models>`.

LogProbQueueModel
~~~~~~~~~~~~~~~~~

..  code-block:: python

    from hftbacktest import LogProbQueueModel

    hbt = HftBacktest(
        data,
        tick_size=0.01,
        lot_size=0.001,
        maker_fee=-0.00005,
        taker_fee=0.0007,
        order_latency=IntpOrderLatency(latency_data),
        queue_model=LogProbQueueModel()
        asset_type=Linear
    )

IdentityProbQueueModel
~~~~~~~~~~~~~~~~~~~~~~

..  code-block:: python

    from hftbacktest import IdentityProbQueueModel

    hbt = HftBacktest(
        data,
        tick_size=0.01,
        lot_size=0.001,
        maker_fee=-0.00005,
        taker_fee=0.0007,
        order_latency=IntpOrderLatency(latency_data),
        queue_model=IdentityProbQueueModel()
        asset_type=Linear
    )

SquareProbQueueModel
~~~~~~~~~~~~~~~~~~~~

..  code-block:: python

    from hftbacktest import SquareProbQueueModel

    hbt = HftBacktest(
        data,
        tick_size=0.01,
        lot_size=0.001,
        maker_fee=-0.00005,
        taker_fee=0.0007,
        order_latency=IntpOrderLatency(latency_data),
        queue_model=SquareProbQueueModel()
        asset_type=Linear
    )

PowerProbQueueModel
~~~~~~~~~~~~~~~~~~~

..  code-block:: python

    from hftbacktest import PowerProbQueueModel

    hbt = HftBacktest(
        data,
        tick_size=0.01,
        lot_size=0.001,
        maker_fee=-0.00005,
        taker_fee=0.0007,
        order_latency=IntpOrderLatency(latency_data),
        queue_model=PowerProbQueueModel(3)
        asset_type=Linear
    )

ProbQueueModel2
---------------
This model is a variation of the `ProbQueueModel`_ that changes the probability calculation to
f(back) / f(front + back) from f(back) / (f(front) + f(back)).

LogProbQueueModel2
~~~~~~~~~~~~~~~~~~

..  code-block:: python

    from hftbacktest import LogProbQueueModel2

    hbt = HftBacktest(
        data,
        tick_size=0.01,
        lot_size=0.001,
        maker_fee=-0.00005,
        taker_fee=0.0007,
        order_latency=IntpOrderLatency(latency_data),
        queue_model=LogProbQueueModel2()
        asset_type=Linear
    )

PowerProbQueueModel2
~~~~~~~~~~~~~~~~~~~~

..  code-block:: python

    from hftbacktest import PowerProbQueueModel2

    hbt = HftBacktest(
        data,
        tick_size=0.01,
        lot_size=0.001,
        maker_fee=-0.00005,
        taker_fee=0.0007,
        order_latency=IntpOrderLatency(latency_data),
        queue_model=PowerProbQueueModel2(3)
        asset_type=Linear
    )

ProbQueueModel3
---------------
This model is a variation of the `ProbQueueModel`_ that changes the probability calculation to
1 - f(front / (front + back)) from f(back) / (f(front) + f(back)).

PowerProbQueueModel3
~~~~~~~~~~~~~~~~~~~~

..  code-block:: python

    from hftbacktest import PowerProbQueueModel3

    hbt = HftBacktest(
        data,
        tick_size=0.01,
        lot_size=0.001,
        maker_fee=-0.00005,
        taker_fee=0.0007,
        order_latency=IntpOrderLatency(latency_data),
        queue_model=PowerProbQueueModel3(3)
        asset_type=Linear
    )

Implement a custom probability queue position model
---------------------------------------------------

.. code-block:: python

    @jitclass
    class CustomProbQueueModel(ProbQueueModel):
        def f(self, x):
            # todo: custom formula
            return x ** 3


Implement a custom queue model
------------------------------
You need to implement ``numba`` ``jitclass`` that has four methods: ``new``, ``trade``, ``depth``, ``is_filled``

See `Queue position model implementation
<https://github.com/nkaz001/hftbacktest/blob/master/hftbacktest/models/queue.py>`_ in detail.

.. code-block:: python

    @jitclass
    class CustomQueuePositionModel:
        def __init__(self):
            pass

        def new(self, order, proc):
            # todo: when a new order is submitted.
            pass

        def trade(self, order, qty, proc):
            # todo: when a trade happens.
            pass

        def depth(self, order, prev_qty, new_qty, proc):
            # todo: when the order book quantity at the price is changed.
            pass

        def is_filled(self, order, proc):
            # todo: check if a given order is filled.
            return False

        def reset(self):
            pass

References
==========
This is initially implemented as described in the following articles.

* http://www.math.ualberta.ca/~cfrei/PIMS/Almgren5.pdf
* https://quant.stackexchange.com/questions/3782/how-do-we-estimate-position-of-our-order-in-order-book
