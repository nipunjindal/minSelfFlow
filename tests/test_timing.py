import time

from minselfflow.utils.timing import segment_timer


def test_segment_timer():
    with segment_timer("Test segment"):
        time.sleep(0.01)
