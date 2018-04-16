import pymysql
import os
import json

connection = pymysql.connect(host='35.231.155.194', user='eplu', passwd='Acan0fBEEZ7683@', db='steam')

cur = connection.cursor()
with open('subset2.txt') as f:
    s = f.read().split()[1:]

d = {}

for r, i in enumerate(s):
    if r % 1000 == 0:
        print(r)
    cur.execute('SELECT * FROM Games_2 WHERE steamid = %s' % i)
    query = cur.fetchall()
    q = []
    for j in query:
        if j[3] is not None and j[3] != 0:
            q.append([j[1], j[3]])
    d[i] = q
    #print q

#print d

with open('user_games2.json', 'w') as f:
    json.dump(d, f, separators = (',', ':'))
