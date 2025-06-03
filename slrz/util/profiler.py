from __future__ import annotations

from typing import Any
import time
from collections import defaultdict, OrderedDict

# Optional import
try:
    # We only import tqdm here to check whether it is installed
    from tqdm import tqdm
except ImportError:
    tqdm = None


class Profiler:
    """
    Simple profiler utility. It can be attached to a tqdm progress bar to display
    live running time statistics.

    ## Usage
    Create a slider wrapping a tqdm progress bar.
    ```python
    pbar = tqdm(range(1000))
    profiler = Profiler(pbar, decay=0.95)
    ```
    Alternatively, you may attach an existing profiler to a tqdm progress bar:
    ```python
    profiler.attach_tqdm(pbar)
    ```

    Pass the profiler to the code you want to profile (usually within a tqdm loop),
    and wrap sections of code you want to measure with a labeled `with` statement:
    ```python
    with tqdm(range(10000), desc="Profiler example") as pbar:
        profiler = Profiler(pbar, decay=0.95)
        sum = 0
        prod = 1
        for i in pbar:
            with profiler['add']:
                for j in range(1000000):
                    sum += j
            with profiler['prod']:
                for j in range(1000000):
                    prod *= j
    ```
    The example above will display a progress bar similar to the following:
    ```
    Profiler example: 10%|█         | 101/10000 [00:01<00:09, 101it/s, add=0.00221s, prod=0.00772s]
    ```
    Section labels need to be unique. Sections may contain other sections.

    ## Overhead
    Naturally, profiling code introduces a slight overhead, involved in updating the running
    metrics, and the time spent by tqdm formatting the statistics for display.

    An alternative to stripping profiled code of profiler references to maximize performance
    after profiling, it is possible to bypass almost all the overhead in instrumented code by
    passing `Profiler.noop` as the profiler.
    This is useful for code in constant evolution, where profiling is a recurrent task,
    but it is also desired to run the code at full speed to perform other tests.

    `Profiler.noop` simply executes its `with` statements without any overhead beyond the
    function calls these blocks entail (`__getitem__`, `__enter__` and `__exit__`).
    """
    noop: Profiler = None

    def __init__(self, tqdm_instance=None, *, decay: float = 0.95, time_format: str = '{value:.5f}s'):
        self.timings: dict[str, float] = OrderedDict()
        self.values: dict[str, Any] = OrderedDict()
        self.counts: dict[str, int] = defaultdict(int)
        self.decay = decay
        self.time_format = time_format
        self._start_times: dict[str, float] = {}

        # Only record the argument as a tqdm instance if tqdm is installed
        # Otherwise, ignore it, to allow code to optionally wrap iterables
        # only when tqdm is installed.
        if tqdm:
            self.tqdm_instance = tqdm_instance
        else:
            self.tqdm_instance = None

    def attach_tqdm(self, tqdm_instance):
        """
        Attach a tqdm progress bar to the profiler.
        """
        if tqdm:
            self.tqdm_instance = tqdm_instance

    def record(self, label: str, value: Any) -> None:
        """
        Record a custom (non-time) statistic for a given label.
        Non-time statistics are not merged/averaged.
        Only the last value is displayed.
        """
        if value is not None:
            self.values[label] = value
        else:
            del self.values[label]

    def start(self, label: str) -> None:
        """
        Start timing a section with the given label.
        """
        self._start_times[label] = time.time()

    def stop(self, label: str) -> float:
        """
        Stop timing a section and update the average.
        """
        if label not in self._start_times:
            raise ValueError(f"No timer started for label '{label}'")

        elapsed = time.time() - self._start_times.pop(label)

        # Update running average with decay
        if self.counts[label] == 0:
            self.timings[label] = elapsed
        else:
            if label not in self.timings:
                self.timings[label] = elapsed
            else:
                self.timings[label] = self.timings[label] * self.decay + elapsed * (1 - self.decay)
        self.counts[label] += 1

        # Update progress bar
        if self.tqdm_instance is not None:
            self._update_tqdm()
        return elapsed

    def _update_tqdm(self) -> None:
        postfix_dict = OrderedDict([
            (label, self.time_format.format(value=value))
            for label, value in self.timings.items()
        ] + [
            (label, value)
            for label, value in self.values.items()
        ])

        self.tqdm_instance.set_postfix(postfix_dict)

    def __getitem__(self, label: str | tuple[str, Any]) -> ProfilerSectionContext | None:
        """
        If one parameter is passed (a string), return a context manager
        for timing a `with` block.

        If two parameters are passed (as a tuple), record a custom
        (non-time) statistic (shortcut for `record(label, value)`).
        """
        if isinstance(label, tuple):
            assert len(label) == 2, "Expected a tuple of length 2"
            label, value = label
            self.record(label, value)
            return None
        return ProfilerSectionContext(self, label)


class ProfilerSectionContext:
    """
    Context manager that times the execution of a `with` statement.
    Calls `profiler.start` and `profiler.stop` internally.
    """
    def __init__(self, profiler: Profiler, label: str):
        self.profiler = profiler
        self.label = label

    def __enter__(self):
        self.profiler.start(self.label)
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.stop(self.label)
        return False


class NoopContext:
    """
    Context manager that executes its `with` statements without any overhead.
    """
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # Do not mark any exceptions as handled


class NoopProfiler(Profiler):
    """
    Profiler that does not perform any profiling.
    Minimizes overhead of instrumented code.
    """
    def __init__(self):
        super().__init__()
        # Create NoopContext instance early to avoid creating it at runtime
        self.context = NoopContext()

    def attach_tqdm(self, tqdm_instance):
        pass
    def record(self, label: str, value: Any) -> None:
        pass
    def start(self, label: str) -> None:
        pass
    def stop(self, label: str) -> float:
        pass
    def _update_tqdm(self) -> None:
        pass

    def __getitem__(self, label: str):
        return self.context


# Initialize the noop profiler
Profiler.noop = NoopProfiler()
