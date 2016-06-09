from time import strftime, localtime


def info(msg):
  ts = strftime("%Y-%m-%d %H:%M:%S", localtime()) + ' :: '
  print ts + msg
  with open('output.log', 'a+') as f:
    f.write(ts + msg + '\n')

