import os

names = open('actresses_2.txt', 'r').read().split('\n')

for name in names:
  os.system('python icrawler_keyword.py "' + name + '"')