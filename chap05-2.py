
newjeans = ['민지', '하니', '다니엘', '해린', '혜인']

# 슬라이싱, 이상 ~ 미만 개념
print(newjeans[1:4]) # 1번 이상 4번 미만
# 원본은 안 바뀜, 슬라이싱은 복사해서 수행
print(newjeans)
print(newjeans[:3]) # 0번부터 3번 미만
print(newjeans[2:]) # 2번부터 끝까지

print('=========')
# 리스트의 끝 인덱스: -1, 첫 인덱스: -(리스트 길이)
print(newjeans[4] == newjeans[-1])
print(f'뉴진스의 리더는 {newjeans[-5]}입니다.')
print(newjeans[-3:])
print(newjeans[:-3])
print(newjeans[1:5:2]) # 1시작, +2씩, 5번째 인덱스 전까지 보여줌

# 특이한 점, 슬라이싱은 리스트의 범위를 넘어가도 에러를 발생시키지 않음
print(newjeans[1:100:2]) 