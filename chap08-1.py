
# 2차원 리스트
# 바깥은 2차원, 안쪽은 1차원

# 한 학생의 국영수 점수를 1차원리스트로 표현
student_yujin = [95, 100, 88] # 안유진 학생의 국영수 점수
student_karina = [92, 94, 98] # 카리나 학생의 국영수 점수
student_chaewon = [88, 90, 92] # 김채원 학생의 국영수 점수

# 이 3명의 학생의 성적들을 하나의 변수로 관리
gradebook = [
    [95, 100, 88],
    [92, 94, 98],
    [88, 90, 92],
    [100, 100, 100]
]

print(gradebook[0])
print(len(gradebook))
print(gradebook[1][1])

print('===================')

seat_map = [
    ['O', 'O', 'O', 'X'],  # A열
    ['O', 'X', 'O', 'O'],  # B열
    ['X', 'O', 'X', 'O'],  # C열
]

# B3좌석을 예매하고싶음
if seat_map[1][2] == 'O':
    print('해당 좌석은 예매 가능합니다.')
else:
    print('이미 예매된 좌석입니다.')

# A4열이 안유진이라는 사람에 의해 예매됨
seat_map[0][3] = '안유진'
print(seat_map)

print('===================')
# 행의 개수
print(len(seat_map))  # 결과: 3
# 1번 행의 열의 개수
print(len(seat_map[1]))  # 결과: 4

print('===================')

count = 0
# 중첩 반복문
for seat in seat_map:
    for n in seat:
        print(n, end=' ')
        if n == 'O':
            count += 1
    print()

print(f'빈 좌석 개수: {count}')

print('===================')

                # [0, 1, 2, 3]
for index in range(len(gradebook)):
    # print(gradebook[index])
    student_scores = gradebook[index]
    total = sum(student_scores)
    avg = total / len(student_scores)

    print(f'{index + 1}번 학생의 총점: {total}점, 평균: {avg: .2f}점')

print('===================')

# 과목별 총점 구하기, 딕셔너리, 열의 합계
subject_total = {
    '국어총점': 0,
    '영어총점': 0,
    '수학총점': 0,
}

for scores in gradebook:
    subject_total['국어총점'] += scores[0]
    subject_total['영어총점'] += scores[1]
    subject_total['수학총점'] += scores[2]

print(subject_total)


