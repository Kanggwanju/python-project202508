
# 6강 복습 퀴즈

# 1번, 딕셔너리
trainee = {
    '안유진': 150,
    '카리나': 200,
    '이서': 100
}
trainee['카리나'] -= 50
trainee['이서'] += 100
trainee['장원영'] = 300 # 값 추가
del trainee['안유진']

print(trainee['이서'])
print(trainee)
print('===================')

a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7}
c = set() # 빈 집합, 셋
c.update(a - b) # {1, 2}
c.update(b - a) # {1, 2, 6, 7}
c.add(len(a & b)) # {1, 2, 6, 7, 3}
print(c)