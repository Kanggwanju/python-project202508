
# 셋, Set
# 중복 제거, 순서 없음

food_list = ['마라탕', '탕후루', '마라탕']

food_set = set(food_list)

print(type(food_set))
print(food_set)

# 딕셔너리 만들기
example_dict = {'a': 1, 'b': 2}
print(type(example_dict))

# set 만들기 (중괄호, ':' 제외)
example_set = {'a', 'b'}
print(type(example_set))

empty_dict = {} # 빈 딕셔너리
empty_set = set() # 빈 셋

print('=====================')

aespa_members = {'카리나', '원터'}
print(f'초기 멤버: {aespa_members}')

# add, set에 데이터 추가
# 중복 제거이므로 같은 데이터 넣어도 1개만 존재
aespa_members.add('닝닝')
aespa_members.add('닝닝')
aespa_members.add('닝닝')
print(aespa_members)

# update, set에 데이터 일괄 추가
aespa_members.update(['지젤', '닝닝', '나비스'])
print(aespa_members)

# 순서 없음에 주의

print('=====================')

jennie_likes = {'딸기', '샤인머스캣', '복숭아'}
karina_likes = {'샤인머스캣', '수박', '망고'}

# 교집합
print(f'둘 다 좋아하는 것: {jennie_likes & karina_likes}')
print(f'둘 다 좋아하는 것: {jennie_likes.intersection(karina_likes)}')

# 합집합
print(f'좋아하는 과일 모음: {jennie_likes | karina_likes}')
print(f'좋아하는 과일 모음: {jennie_likes.union(karina_likes)}')

# 차집합
print(f'제니만 좋아하는 것: {jennie_likes - karina_likes}')
print(f'제니만 좋아하는 것: {jennie_likes.difference(karina_likes)}')
