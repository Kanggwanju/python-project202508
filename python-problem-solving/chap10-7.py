# 기출 풀이 7. 문자열 슬라이싱, % formatting

a = "REMEMBER NOVEMBER"
b = a[:3] + a[12:15]
c = "R AND %s" % "STR"

print(b + c)