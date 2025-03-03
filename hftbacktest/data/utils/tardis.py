import gzip
from typing import List, Optional, Literal

import numpy as np
from numpy.typing import NDArray

from .. import merge_on_local_timestamp, correct, validate_data
from ... import DEPTH_CLEAR_EVENT, DEPTH_SNAPSHOT_EVENT, TRADE_EVENT, DEPTH_EVENT


def convert(
        input_files: List[str],
        output_filename: Optional[str] = None,
        buffer_size: int = 100_000_000,
        ss_buffer_size: int = 10_000,
        base_latency: float = 0,
        method: Literal['separate', 'adjust'] = 'separate',
        snapshot_mode: Literal['process', 'ignore_sod', 'ignore'] = 'process'
) -> NDArray:
    """
    Converts Tardis.dev data files into a format compatible with HftBacktest.

    Args:
        input_files (List[str]): Input filenames for both incremental book and trades files.
        output_filename (Optional[str], optional): If provided, the converted data will be saved to the specified filename in npz format. Defaults to None.
        buffer_size (int, optional): Sets a preallocated row size for the buffer. Defaults to 100_000_000.
        ss_buffer_size (int, optional): Sets a preallocated row size for the snapshot buffer. Defaults to 10_000.
        base_latency (float, optional): The value to be added to the feed latency. Defaults to 0.
        method (Literal['separate', 'adjust'], optional): The method to correct reversed exchange timestamp events. Defaults to 'separate'.
        snapshot_mode (Literal['process', 'ignore_sod', 'ignore'], optional): The mode for processing snapshot events. Defaults to 'process'.

    Returns:
        NDArray: Converted data compatible with HftBacktest.
    """
    # Function code here
def convert(
        input_files: List[str],
        output_filename: Optional[str] = None,
        buffer_size: int = 100_000_000,
        ss_buffer_size: int = 10_000,
        base_latency: float = 0,
        method: Literal['separate', 'adjust'] = 'separate',
        snapshot_mode: Literal['process', 'ignore_sod', 'ignore'] = 'process'
) -> NDArray:
    r"""
    Converts Tardis.dev data files into a format compatible with HftBacktest.

    Args:
        input_files: Input filenames for both incremental book and trades files,
                     e.g. ['incremental_book.csv', 'trades.csv'].
        output_filename: If provided, the converted data will be saved to the specified filename in ``npz`` format.
        buffer_size: Sets a preallocated row size for the buffer.
        base_latency: The value to be added to the feed latency.
                      See :func:`.correct_local_timestamp`.
        method: The method to correct reversed exchange timestamp events. See :func:`..validation.correct`.
        snapshot_mode: - If this is set to 'ignore', all snapshots are ignored. The order book will converge to a
                         complete order book over time.
                       - If this is set to 'ignore_sod', the SOD (Start of Day) snapshot is ignored.
                         Since Tardis intentionally adds the SOD snapshot, not due to a message ID gap or disconnection,
                         there might not be a need to process SOD snapshot to build a complete order book.
                         Please see https://docs.tardis.dev/historical-data-details#collected-order-book-data-details
                         for more details.
                       - Otherwise, all snapshot events will be processed.

    Returns:
        Converted data compatible with HftBacktest.
    """
    TRADE = 0
    DEPTH = 1

    sets = []
    for file in input_files:
        file_type = None
        tmp = np.empty((buffer_size, 6), np.float64)
        row_num = 0
        is_snapshot = False
        ss_bid = None
        ss_ask = None
        ss_bid_rn = 0
        ss_ask_rn = 0
        is_sod_snapshot = True
        print('Reading %s' % file)
        with gzip.open(file, 'r') as f:
            while True:
                line = f.readline()
                if line is None or line == b'':
                    break
                cols = line.decode().strip().split(',')
                if len(cols) < 8:
                    print('Warning: Invalid Data Row', cols, line)
                    continue
                if file_type is None:
                    if cols == [
                        'exchange',
                        'symbol',
                        'timestamp',
                        'local_timestamp',
                        'id',
                        'side',
                        'price',
                        'amount'
                    ]:
                        file_type = TRADE
                    elif cols == [
                        'exchange',
                        'symbol',
                        'timestamp',
                        'local_timestamp',
                        'is_snapshot',
                        'side',
                        'price',
                        'amount'
                    ]:
                        file_type = DEPTH
                elif file_type == TRADE:
                    # Insert TRADE_EVENT
                    tmp[row_num] = [
                        TRADE_EVENT,
                        int(cols[2]),
                        int(cols[3]),
                        1 if cols[5] == 'buy' else -1,
                        float(cols[6]),
                        float(cols[7])
                    ]
                    row_num += 1
                elif file_type == DEPTH:
                    if cols[4] == 'true':
                        if (snapshot_mode == 'ignore') or (snapshot_mode == 'ignore_sod' and is_sod_snapshot):
                            continue
                        # Prepare to insert DEPTH_SNAPSHOT_EVENT
                        if not is_snapshot:
                            is_snapshot = True
                            ss_bid = np.empty((ss_buffer_size, 6), np.float64)
                            ss_ask = np.empty((ss_buffer_size, 6), np.float64)
                            ss_bid_rn = 0
                            ss_ask_rn = 0
                        if cols[5] == 'bid':
                            ss_bid[ss_bid_rn] = [
                                DEPTH_SNAPSHOT_EVENT,
                                int(cols[2]),
                                int(cols[3]),
                                1,
                                float(cols[6]),
                                float(cols[7])
                            ]
                            ss_bid_rn += 1
                        else:
                            ss_ask[ss_ask_rn] = [
                                DEPTH_SNAPSHOT_EVENT,
                                int(cols[2]),
                                int(cols[3]),
                                -1,
                                float(cols[6]),
                                float(cols[7])
                            ]
                            ss_ask_rn += 1
                    else:
                        is_sod_snapshot = False
                        if is_snapshot:
                            # End of the snapshot.
                            is_snapshot = False

                            # Add DEPTH_CLEAR_EVENT before refreshing the market depth by the snapshot.
                            ss_bid = ss_bid[:ss_bid_rn]
                            if len(ss_bid) > 0:
                                # Clear the bid market depth within the snapshot bid range.
                                tmp[row_num] = [
                                    DEPTH_CLEAR_EVENT,
                                    ss_bid[0, 1],
                                    ss_bid[0, 2],
                                    1,
                                    ss_bid[-1, 4],
                                    0
                                ]
                                row_num += 1
                                # Add DEPTH_SNAPSHOT_EVENT for the bid snapshot
                                tmp[row_num:row_num + len(ss_bid)] = ss_bid[:]
                                row_num += len(ss_bid)
                            ss_bid = None

                            ss_ask = ss_ask[:ss_ask_rn]
                            if len(ss_ask) > 0:
                                # Clear the ask market depth within the snapshot ask range.
                                tmp[row_num] = [
                                    DEPTH_CLEAR_EVENT,
                                    ss_ask[0, 1],
                                    ss_ask[0, 2],
                                    -1,
                                    ss_ask[-1, 4],
                                    0
                                ]
                                row_num += 1
                                # Add DEPTH_SNAPSHOT_EVENT for the ask snapshot
                                tmp[row_num:row_num + len(ss_ask)] = ss_ask[:]
                                row_num += len(ss_ask)
                            ss_ask = None
                        # Insert DEPTH_EVENT
                        tmp[row_num] = [
                            DEPTH_EVENT,
                            int(cols[2]),
                            int(cols[3]),
                            1 if cols[5] == 'bid' else -1,
                            float(cols[6]),
                            float(cols[7])
                        ]
                        row_num += 1
        sets.append(tmp[:row_num])

    print('Merging')
    data = sets[0]
    del sets[0]
    while len(sets) > 0:
        data = merge_on_local_timestamp(data, sets[0])
        del sets[0]

    data = data[np.argsort(data[:, 2])]
    data = correct(data, base_latency=base_latency, method=method)

    # Validate again.
    num_corr = validate_data(data)
    if num_corr < 0:
        raise ValueError

    if output_filename is not None:
        print('Saving to %s' % output_filename)
        np.savez(output_filename, data=data)

    return data
