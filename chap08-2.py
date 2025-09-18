# for 문
# squares = []
# for i in range(1, 6):
#     squares.append(i * i)

# lambda, 람다
# squares = list(map(lambda i: i * i, range(1, 6)))

# 리스트 컴프리헨션
squares = [i * i for i in range(1, 6)]

print(squares)

print('=======================')

# even_numbers = []
# for i in range(1, 11):
#     if i % 2 == 0:
#         even_numbers.append(i)

# even_numbers = list(filter(lambda i: i % 2 == 0, range(1, 11)))

even_numbers = [i for i in range(1, 11) if i % 2 == 0]

print(even_numbers)