# 기출 풀이 3. 클래스, 슬라이싱

class Good:
    li = ["seoul", "kyeonggi", "incheon", "daejeon", "daegu", "pusan"]

g = Good()
str01 = ''

for i in g.li:
    str01 = str01 + i[0]

print(str01)

