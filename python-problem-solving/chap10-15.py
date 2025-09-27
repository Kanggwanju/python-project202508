# 기출 풀이 15. 트리, 클래스, 리스트 컴프리헨션, range, 리스트, append
# 조건부 표현식, 재귀

class Node:
    def __init__(self, v):
        self.v=v;
        self.children=[]

def tree(li):
    n=[Node(i) for i in li] # 리스트 컴프리헨션
    for i in range(1, len(li)):
        n[(i-1)//2].children.append(n[i])
    return n[0]

def calc(node, level=0):
    if node is None:
        return 0

    # 조건부 표현식 + 리스트 컴프리헨션 + 재귀 호출
    # level이 홀수면 node.v, 짝수면 0
    return (node.v if level % 2 == 1 else 0) + sum(calc(i, level+1) for i in node.children)

root = tree([3, 5, 8, 12, 15, 18, 21])
print(calc(root))