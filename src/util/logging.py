
def info(msg):
  print msg
  with open('output.log', 'a+') as f:
    f.write(msg + '\n')

