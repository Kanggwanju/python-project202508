
for i in range(10, 1, -2):
    print(i, end=' ')

print('\n========')

level = 1
exp = 0
while exp < 100:
    exp = exp + (level * 15)
    level = level + 1
print(level)

print('========')

for i in range(1, 6):
    if i == 3:
        continue
    print(f'{i}반, 대청소 시작!')

print('========')

fruits = ["딸기", "귤", "포도", "파인애플", "샤인머스캣", "블루베리"]
count = 0
for fruit in fruits:
    if fruit == "샤인머스캣":
        break
    count = count + 1
print(count)

