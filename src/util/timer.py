import time


def _fmt_time(t):
  return '%dh %dmin %ds %dms' % (t // (60 * 60 * 1000), t // (60 * 1000), t // 1000, t % 1000)


class Timer:
  def __init__(self):
    self.last  = 0
    self.total = 0

  def start(self):
    self.last = int(round(time.time() * 1000))

  def lap(self):
    t = int(round(time.time() * 1000))
    d = t - self.last
    self.last = t
    self.total += d
    return _fmt_time(d), _fmt_time(self.total)
