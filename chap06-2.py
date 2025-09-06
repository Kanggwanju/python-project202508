
# 딕셔너리, 사전, dict
# 3.7 버전 이상 딕셔너리에 요소를 추가한 순서가 그대로 유지
wonyoung_profile = {
    '이름': '장원영',
    '그룹': '아이브',
    '포지션': '센터',
    '출생': 2004,
    'MBTI': 'ESFP',
}

print(type(wonyoung_profile))
print(wonyoung_profile)

print('=====================')

print(f"이름이 뭐에요? {wonyoung_profile['이름']}")

# 없는 키를 참조할 경우 에러 발생, if문으로 검사 필수
# 안전 코딩법
if "MBTI" in wonyoung_profile:
    print(f"MBTI가 뭐에요? {wonyoung_profile['MBTI']}")
else:
    print('해당 키는 존재하지 않습니다.')

print('=====================')

# get을 쓸때 없으면 None, 안전
print(wonyoung_profile.get('신장', '해당 키는 존재하지 않습니다.'))

print('=====================')

# 딕셔너리에 값 추가하기 (없는 키를 사용)
wonyoung_profile['별명'] = '갓기'

# 딕셔너리에 값 수정하기 (있는 키를 사용, 에러 발생 안함)
wonyoung_profile['MBTI'] = 'IM17'

# 딕셔너리에 값 삭제하기 (있는 키를 사용, 에러 발생 가능) 
del wonyoung_profile['포지션']
print(wonyoung_profile)

print('=====================')

# 딕셔너리의 키만 추출
print(wonyoung_profile.keys())

# 딕셔너리의 value만 추출
print(wonyoung_profile.values())

# 모든 키와 value를 튜플!로 묶어서 반환
print(wonyoung_profile.items())

print('=====================')

# key만 꺼내서 반복
for key in wonyoung_profile.keys():
    value = wonyoung_profile[key]
    print(f'key: {key}, value: {value}')

print('=====================')

# items로 반복
for entry in wonyoung_profile.items():
    # print(entry)
    print(f'key: {entry[0]}, value: {entry[1]}')

for key, value in wonyoung_profile.items():
    print(f'key: {key}, value: {value}')