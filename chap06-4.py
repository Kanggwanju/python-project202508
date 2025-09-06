
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

# 2번, 집합, Set
a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7}
c = set() # 빈 집합, 셋
c.update(a - b) # {1, 2}
c.update(b - a) # {1, 2, 6, 7}
c.add(len(a & b)) # {1, 2, 6, 7, 3}
print(c)
print('===================')

# 4번, 리스트, 딕셔너리, 셋
records = [
    {'group': '아일릿', 'user': '민주팬'},
    {'group': '르세라핌', 'user': '채원팬'},
    {'group': '아일릿', 'user': '원희팬'},
    {'group': '아일릿', 'user': '민주팬'},
    {'group': '르세라핌', 'user': '카즈하팬'},
    {'group': '아일릿', 'user': '윤아팬'}
]
application_db = {} # 빈 딕셔너리

for rec in records: # rec는 딕셔너리
    group_name = rec['group'] # 아일릿, 르세라핌
    user_id = rec['user'] # 민주팬, 채원팬, ...

    if group_name not in application_db: # 그룹 이름 키가 없을 경우
        application_db[group_name] = set() # '그룹 이름': '빈 셋'({}) 생성
    
    application_db[group_name].add(user_id) # 그룹 이름이 키인 value('Set')에 팬 이름 넣음

print(len(application_db['아일릿']))
print(application_db)
print('===================')

# 5번 리스트, 딕셔너리, 셋
sales = ['딸기', '샤인머스캣', '딸기', '귤', '샤인머스캣', '딸기']
sales_report = {}
top_sellers = set()

for fruit in sales:
    if fruit not in sales_report:
        sales_report[fruit] = 0
    sales_report[fruit] += 1
print(sales_report)

for fruit_name, count in sales_report.items():
    if count >= 2:
        top_sellers.add(fruit_name)

print(top_sellers)