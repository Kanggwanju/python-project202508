
# def add(n1, n2):
#     return n1 + n2

# 람다함수 (매개변수: return)
# 일회용 함수
add = lambda n1, n2: n1 + n2

# def pow(n):
#     return n ** 2

pow = lambda n: n ** 2

print(add(10, 20))
print(pow(10))

print('===================')

numbers = [1, 2, 3, 4, 5, 6]

# double_numbers = []
# for n in numbers:
#     double_numbers.append(n * 2)

# map(적용할 함수, 데이터 뭉치), 개수는 그대로, 내용물만 변형
double_numbers = list(map(lambda n: n * 2, numbers))

print(double_numbers)

print('===================')

# even_numbers = []
# for n in numbers:
#     if n % 2 == 0:
#         even_numbers.append(n)

# filter(적용할 함수, 데이터 뭉치), 내용물 그대로, 조건에 맞는 것만 선별
even_numbers = list(filter(lambda n: n % 2 == 1, numbers))

print(even_numbers)