import numpy as np
from multiprocessing import Array
from ctypes import c_float


def create_shared_memory(n_rows: int, n_cols: int):
    """Create shared memory arrays for cross-process portfolio tracking.

    Returns a dict of multiprocessing.Array objects that can be shared
    between ProcessPoolExecutor workers via init_worker.
    """
    shared_arrays = {
        'port_begin_balance': Array(c_float, n_rows),
        'port_end_balance': Array(c_float, n_rows * n_cols),
        'port_equity': Array(c_float, n_rows * n_cols),
        'used_margin': Array(c_float, n_rows * n_cols),
        'free_margin': Array(c_float, n_rows)
    }

    for key, array in shared_arrays.items():
        np_array = np.frombuffer(array.get_obj(), dtype=np.float32)
        if key in ['port_end_balance', 'port_equity', 'used_margin']:
            np_array = np_array.reshape((n_rows, n_cols))
        np_array.fill(np.nan)

    return shared_arrays


class SharedMemoryManager:
    """Lazy wrapper around shared memory arrays.

    Reconstructs numpy views from multiprocessing.Array on first access,
    then caches them for the lifetime of the manager.
    """

    def __init__(self, temp_port_record, n_rows):
        self.temp_port_record = temp_port_record
        self.n_rows = n_rows
        self._portfolio_arrays = None

    @property
    def portfolio_arrays(self):
        if self._portfolio_arrays is None:
            self._portfolio_arrays = self._reconstruct_arrays
        return self._portfolio_arrays

    @property
    def _reconstruct_arrays(self):
        return {
            'port_begin_balance': np.frombuffer(
                self.temp_port_record['port_begin_balance'].get_obj(), dtype=np.float32),
            'port_end_balance': np.frombuffer(
                self.temp_port_record['port_end_balance'].get_obj(), dtype=np.float32).reshape((self.n_rows, -1)),
            'port_equity': np.frombuffer(
                self.temp_port_record['port_equity'].get_obj(), dtype=np.float32).reshape((self.n_rows, -1)),
            'used_margin': np.frombuffer(
                self.temp_port_record['used_margin'].get_obj(), dtype=np.float32).reshape((self.n_rows, -1)),
            'free_margin': np.frombuffer(
                self.temp_port_record['free_margin'].get_obj(), dtype=np.float32)
        }
