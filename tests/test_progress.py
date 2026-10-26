from ariostea.domain.models import Progress
from ariostea.progress import ProgressPrinter


class Clock:
    def __init__(self, step=0.0):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def _feed(printer, total, embedded=0, skipped=0):
    for done in range(1, total + 1):
        printer(Progress(done=done, total=total, embedded=embedded, skipped=skipped))


def test_printer_prints_every_100_and_at_end():
    lines = []
    _feed(ProgressPrinter(lines.append, clock=Clock()), 250)
    assert [line.split(" ")[0] for line in lines] == ["100/250", "200/250", "250/250"]


def test_printer_prints_on_interval():
    lines = []
    _feed(ProgressPrinter(lines.append, clock=Clock(step=10.0)), 7)
    assert len(lines) == 7


def test_printer_waits_for_the_interval():
    lines = []
    _feed(ProgressPrinter(lines.append, clock=Clock(step=4.0)), 7)
    # The clock reads 4 s at start and 4 s more per call: 12 s by call 3, then 6.
    assert [line.split(" ")[0] for line in lines] == ["3/7", "6/7", "7/7"]


def test_printer_format():
    lines = []
    printer = ProgressPrinter(lines.append, clock=Clock())
    printer(Progress(done=400, total=3410, embedded=37, skipped=2))
    assert lines == ["400/3,410 notes · 37 embedded · 2 skipped"]
