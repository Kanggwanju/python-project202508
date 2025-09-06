
# 리스트 (멤버가 변경될 수도 있는 자료구조)
project_group = ['유재석', '하하']

# 튜플 (멤버가 절대 변경되지 않는 자료구조)
newjeans = ('민지', '하니', '다니엘', '해린', '혜인')

# 튜플
important_mind = '중요한건', '꺾이지 않는', '마음'

print(type(project_group))
print(type(newjeans))
print(type(important_mind))

print ('=============')

# 튜플 인덱싱은 리스트와 동일
print(f"뉴진스의 센터: {newjeans[0]}")
print(f"뉴진스의 막내: {newjeans[-1]}")
print(f"뉴진스의 언니라인: {newjeans[0:2]}")
print(f'뉴진스의 총: {len(newjeans)}명입니다.')

# 추가 불가능
# newjeans.append('하츄핑')

# 수정 불가능(대입 불가능)
# newjeans[1] = '오로라핑'

# 삭제 불가능(remove 또한 불가)
# newjeans.pop()
