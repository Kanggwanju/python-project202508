# 기출 풀이 5. 튜플, 다중할당, True, False

a, b = 100, 200  # 다중할당

print(a)
print(b)
print(a == b) # True, False 대소문자 조심

x = (10, 20, 30)
print(x)
a = x[0]
b = x[1]
c = x[2]

a, b, c = (10, 20, 30)

# 개수가 똑같아야함 아래는 에러 발생
# a, b = (10, 20, 30)
# a, b, c, d = (10, 20, 30)

