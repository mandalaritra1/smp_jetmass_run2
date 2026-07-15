"""Streaming Dask executor: accumulate each chunk as it completes.

Alternative to coffea's DaskExecutor for the merge-OOM problem: DaskExecutor
tree-reduces chunk outputs *on the workers* (``treereduction=N``), so with
many-axis histograms the partial sums grow until small condor workers die at
the end of the run. Here workers never merge anything -- each finished chunk's
output is pulled back to the client, added in place to the running total, and
its distributed memory is released. Peak merge memory is (running total + one
chunk) on the client, independent of worker memory.

CAVEAT: the client process must still hold the full accumulated output; this
removes the worker-side merge spike only. If the final accumulator itself does
not fit on the submit machine, no executor choice will save you -- trim the
histogram axes / mode instead.

Pattern borrowed from Pepper's ResumableExecutor (pepper/executor.py); this is
the accumulation part only (no state file / resume).

Works with coffea.processor.Runner exactly like DaskExecutor:

    executor = StreamingDaskExecutor(client=client, retries=6, status=True)

Coffea-version caveat: the private helpers imported below
(``_compression_wrapper``, ``_decompress``, accumulator ``iadd``) are verified
against coffea 2026.5.0 (the local venv). When running at LPC on an older
coffea (e.g. 2025.1.0), check these imports still resolve before relying on
this executor there.
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional

from coffea.processor.accumulator import iadd
from coffea.processor.executor import (
    ExecutorBase,
    _compression_wrapper,
    _decompress,
)


def _process_group(function, items):
    """Process a group of work items sequentially on one worker, merging each
    chunk's output into a local partial as it is produced. Only (one chunk's
    events + the partial) are alive at any time, so this cuts the number of
    full-histogram outputs shipped to the client by len(items) without the
    processing-memory cost of a larger chunksize."""
    accumulator = None
    for item in items:
        result = function(item)
        if accumulator is None:
            accumulator = result
        else:
            iadd(accumulator, result)
    return accumulator


@dataclass
class StreamingDaskExecutor(ExecutorBase):
    """worker_merge > 1 groups that many consecutive chunks into one worker
    task (see _process_group). Consecutive chunks usually share a dataset, so
    the merged partial stays chunk-sized and client-bound traffic drops by
    ~worker_merge. Trade-offs: a failed chunk retries its whole group, and the
    per-task wall time scales with worker_merge -- keep n_chunks/worker_merge
    well above the worker count."""

    client: Optional["distributed.Client"] = None  # noqa: F821
    retries: int = 3
    priority: int = 0
    worker_merge: int = 1

    def __getstate__(self):
        # Dask clients are not picklable; drop it if the executor itself ever
        # gets shipped (mirrors coffea's DaskExecutor).
        return dict(self.__dict__, client=None)

    def __call__(self, items, function, accumulator):
        from distributed import as_completed
        from distributed.scheduler import KilledWorker

        items = list(items)
        if len(items) == 0:
            return accumulator, 0

        merge = max(int(self.worker_merge or 1), 1)
        if merge > 1:
            # Group BEFORE the compression wrapper so each chunk output stays
            # uncompressed within the worker and only the merged partial is
            # compressed for the trip back to the client.
            work_items = [items[i:i + merge] for i in range(0, len(items), merge)]
            function = partial(_process_group, function)
        else:
            work_items = items

        if self.compression is not None:
            function = _compression_wrapper(
                self.compression, function, name=self.function_name
            )

        futures = self.client.map(
            function,
            work_items,
            pure=False,
            priority=self.priority,
            retries=self.retries,
        )
        # KilledWorker only carries the task key; keep the mapping so the error
        # can name the actual (file, chunk) work item(s).
        key_to_item = {f.key: item for f, item in zip(futures, work_items)}
        n_chunks = {
            f.key: (len(item) if merge > 1 else 1)
            for f, item in zip(futures, work_items)
        }

        progress = None
        if self.status:
            from tqdm.auto import tqdm

            progress = tqdm(total=len(items), desc=self.desc, unit=self.unit)

        try:
            for future in as_completed(futures):
                try:
                    result = future.result()
                except KilledWorker as ex:
                    raise RuntimeError(
                        f"Work item {key_to_item.get(ex.task, ex.task)} caused "
                        "a KilledWorker exception (likely a segfault or "
                        "out-of-memory issue)"
                    ) from ex
                # Release the distributed copy immediately so worker/scheduler
                # memory stays flat over the run.
                future.release()
                if self.compression is not None:
                    result = _decompress(result)
                if accumulator is None:
                    accumulator = result
                else:
                    iadd(accumulator, result)
                if progress is not None:
                    progress.update(n_chunks[future.key])
        finally:
            if progress is not None:
                progress.close()

        return accumulator, 0
