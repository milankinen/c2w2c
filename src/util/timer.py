import time


def _fmt_time(t):
  hours   = t // (60 * 60 * 1000)
  mins    = (t % (60 * 60 * 1000)) // (60 * 1000)
  secs    = (t % 60 * 1000) // 1000
  millis  = t % 1000
  return '%dh %dmin %ds %dms' % (hours, mins, secs, millis)


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
