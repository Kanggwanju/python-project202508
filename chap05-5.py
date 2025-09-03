
# 복습 퀴즈 1
#           0     |    1          2        3        4
foods = ['마라탕', '탕후루', '제로콜라', '약과', '베이글']
#           -5        -4          -3    |  -2        -1
my_picks = foods[1:-2]
print(my_picks)

# 복습 퀴즈 2
ive = ['안유진', '가을', '레이', '리즈', '이서']
ive.append('장원영')
ive.pop()
ive.insert(1, ive.pop(2))
print(ive)

# 복습 퀴즈 3
numbers = [4, 1, 5, 2, 5, 3]
numbers.remove(5)
numbers.sort()

print(numbers[2])
