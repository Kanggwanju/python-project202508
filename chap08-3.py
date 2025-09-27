
# 문자열 메서드

song_title = "Super Shy"

# 인덱싱
print(song_title[0])

# 슬라이싱
print(song_title[6:])

# split: 특정 규칙에 따라 쪼개서 리스트로 만들어줌
# 문자열이나 규칙에 있는 공백을 조심
my_favorites = "마라탕,탕후루,제로콜라"
food_list = my_favorites.split(',')
print(food_list)

# join: 리스트 안의 문자열들을 하나의 긴 문자열로 합쳐줌
# 문법: '연결자'.join(리스트)
phone_parts = ["010", "1234", "5678"]
full_phone_number = "-".join(phone_parts)
print(full_phone_number)

# replace: 특정 부분을 다른 것으로 교체
s = "파이선 공부"
s = s.replace("파이선", "파이썬")
print(s)

# strip: 양옆의 불필요한 공백/문자 제거
# '아'를 제거하고싶으면 ' 아'를 파라미터로 넣어줘야함.
s2 = " 아이디 "
s2 = s2.strip()
print(s2)

# 양쪽 끝에서부터 . , ! 문자를 하나씩 확인하며 지워나가다가,
# 제거 대상이 아닌 첫 글자를 만나는 순간 작업 중지
text = ".,.,!!Hel.lo!!.,.,."
cleaned_text = text.strip(".,!")
print(cleaned_text)

# lstrip(): 문자열의 왼쪽 끝을 정리
# rstrip(): 문자열의 오른쪽 끝을 정리

# find: 특정 문자가 처음 나타나는 인덱스 찾기 (없으면 -1)
# count: 특정 문자가 총 몇번 나타나는지 세기
lyrics = "Tell me what you want"
print(lyrics.find('what'))
print(lyrics.count('e'))

# upper: 모두 대문자로 변환
# lower: 모두 소문자로 변환
# 원본이 변경되진 않음, 사용자 입력 비교 or 데이터 정규화할 때 필수
mixed_case = "PyThOn Is FuN"
print(mixed_case.upper())
print(mixed_case.lower())
