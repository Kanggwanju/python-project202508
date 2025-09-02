#        [0, 1, 2, 3, 4]
for i in range(5):
    print(f'{i+1}번째 탕후루 만드는중...낼름')

print('========')

print(list(range(70, 90)))

# 1 ~ 10까지의 누적합
total = 0
#           1 ~ 10
for i in range(1, 11):
    total += i
print(f"1부터 10까지의 총합: {total}")

print('========')

# 마지막 파라미터: step, 얼마나 건너뛸지 정함
print(list(range(1, 11, 2)))
print(list(range(2, 11, 2)))

total = 0
#         2, 4, 6, 8, 10
for i in range(2, 11, 2):
    total += i
print(f'짝수의 합: {total}')

print('========')

print(list(range(5, 0, -1)))

# 카운트다운 만들기
for i in range(5, 0, -1):
    print(f'{i}...')
print('발사!!!')

print('========')

for i in range(1, 31):
    if i % 3 == 0:
        print('박수 짝!')
    else:
        print(i)
