import heapq
import threading
import gc


class TaskQueue:
    """Thread-safe priority queue for strategy backtest tasks.

    Tasks are ordered by their starting timestamp so strategies
    are processed in chronological order. Supports requeuing when
    a strategy needs to wait for shared-principal data.
    """

    def __init__(self, tasks):
        self.lock = threading.Lock()
        self.tasks = []
        self.completed_tasks = set()
        self.cleanup_threshold = 20
        self.operation_count = 0

        for task in tasks:
            timestamp_start = next(iter(task[0]))
            strat_id = task[3]
            heapq.heappush(self.tasks, (timestamp_start, strat_id, task))

    def get_task(self):
        with self.lock:
            self._increment_operation_counter()
            if self.tasks:
                return heapq.heappop(self.tasks)[2]
            return None

    def add_task(self, task):
        with self.lock:
            timestamp_start = next(iter(task[0]))
            strat_id = task[3]
            heapq.heappush(self.tasks, (timestamp_start, strat_id, task))
            self._increment_operation_counter()

    def mark_completed(self, strat_id):
        with self.lock:
            self.completed_tasks.add(strat_id)

    def is_empty(self):
        with self.lock:
            return len(self.tasks) == 0

    def size(self):
        with self.lock:
            return len(self.tasks)

    def get_memory_stats(self):
        with self.lock:
            return {
                'queue_size': len(self.tasks),
                'completed_count': len(self.completed_tasks)
            }

    def _increment_operation_counter(self):
        self.operation_count += 1
        if self.operation_count % self.cleanup_threshold == 0:
            gc.collect()
            self._compact_task_queue()

    def _compact_task_queue(self):
        if len(self.tasks) > 0:
            temp_tasks = []
            while self.tasks:
                temp_tasks.append(heapq.heappop(self.tasks))
            self.tasks = []
            for task_tuple in temp_tasks:
                heapq.heappush(self.tasks, task_tuple)
            del temp_tasks


class TaskMemoryManager:
    """Tracks active task references and performs periodic memory cleanup."""

    def __init__(self):
        self.active_tasks = {}
        self.cleanup_frequency = 15
        self.completion_count = 0

    def register_task(self, future, task):
        self.active_tasks[future] = {
            'task': task,
            'strategy_id': task[3],
            'timestamp_start': next(iter(task[0]))
        }

    def complete_task(self, future):
        if future in self.active_tasks:
            task_info = self.active_tasks.pop(future)
            strat_id = task_info['strategy_id']
            self.completion_count += 1
            del task_info['task']
            del task_info

            if self.completion_count % self.cleanup_frequency == 0:
                self._perform_periodic_cleanup()
            return strat_id
        return None

    def _perform_periodic_cleanup(self):
        completed_futures = [f for f in self.active_tasks.keys() if f.done()]
        for future in completed_futures:
            if future in self.active_tasks:
                del self.active_tasks[future]
        for _ in range(3):
            gc.collect()


class BatchCompletionManager:
    """Triggers memory cleanup when batches of strategies finish."""

    def __init__(self, total_strategies, batch_size=10):
        self.total_strategies = total_strategies
        self.batch_size = batch_size
        self.completed_strategies = set()
        self.completed_batches = 0

    def mark_completed(self, strat_id):
        self.completed_strategies.add(strat_id)
        if len(self.completed_strategies) % self.batch_size == 0:
            self._perform_batch_cleanup()
            return True
        return False

    def _perform_batch_cleanup(self):
        self.completed_batches += 1
        for _ in range(3):
            gc.collect()

    def force_final_cleanup(self):
        self.completed_strategies.clear()
        for _ in range(5):
            gc.collect()
