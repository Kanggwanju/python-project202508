# 기출 풀이 12. 리스트, for문, 문자열 인덱싱

a = ["Seoul", "Kyeonggi", "Incheon", "Daejun", "Daegu", "Pusan"]
str = "S"

for i in a:
    str = str + i[1]
print(str)
