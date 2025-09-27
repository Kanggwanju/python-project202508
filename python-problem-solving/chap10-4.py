# 기출 풀이 4. 반복문, range, 비트 이동 연산자

a = 100
result = 0
for i in range(1, 3):
    result = a >> i  # >>: / 2^i, <<: * 2^i
    result = result + 1
print(result)

print('==============')
a = 20
print(a >> 3) # 몫이 나옴
