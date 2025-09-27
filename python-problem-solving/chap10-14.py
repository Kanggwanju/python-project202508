# 기출 풀이 14. 함수, for문, range, sum, 다중할당, 슬라이싱

# 배열 reverse 함수
def func(lst):
    for i in range(len(lst) // 2): # // : 몫
        lst[i], lst[-i-1] = lst[-i-1], lst[i]

lst = [1, 2, 3, 4, 5, 6]
func(lst)
print(lst)

print(sum(lst[::2]) - sum(lst[1::2])) # 1:2:2 -> ::2, 처음부터 끝까지 2개씩 건너뛰며