import json
from pprint import pprint

with open('female_ranker.json') as json_data:
    data = json.load(json_data)

people = data['listItems']

f = open('actresses_2.txt', 'w')
for person in people:
    f.write(person['node']['name'] + '\n')
