# 기출 풀이 2. 2차원 리스트

lol = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
print(lol[0])
print(lol[2][1])

for sub in lol:
    for item in sub:
        print(item, end='') # 줄바꿈 x
    print()