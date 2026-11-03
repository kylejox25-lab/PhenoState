"""Elapsed-time progress on stderr; stdout remains available for result JSON."""

import sys
import time


class Progress:
    def __init__(self, every=50):
        self.every = every
        self.started = time.monotonic()

    def __call__(self, message):
        if self.every:
            print(f"[{time.monotonic() - self.started:8.1f}s] {message}", file=sys.stderr, flush=True)

    def track(self, items, description):
        if not self.every:
            yield from items
            return
        total = len(items)
        self(f"{description}: 0/{total}")
        last_report = time.monotonic()
        for count, item in enumerate(items, 1):
            yield item
            if count == 1 or count == total or count % self.every == 0 or time.monotonic() - last_report >= 30:
                self(f"{description}: {count}/{total}")
                last_report = time.monotonic()
