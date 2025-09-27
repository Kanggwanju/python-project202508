# 기출 풀이 13. 함수, 문자열 슬라이싱, f-string, len
# 문자열이 포함된 개수 찾기

def fnCalculation(x, y):
    result = 0
    for i in range(len(x)):
        if x[i:i+len(y)] == y:
            result += 1
    return result

a = "abdcabcabca"
out = f"ab{fnCalculation(a, 'ab')}ca{fnCalculation(a, 'ca')}"
print(out)