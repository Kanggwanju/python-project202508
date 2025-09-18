
# 8강 복습 퀴즈

# 1번, 2차원 리스트
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
total = 0
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if i == j:
            total += matrix[i][j]
print(total)

print("======================")

# 2번, 문자열, 딕셔너리
raw_data = "  IDOL : NewJeans, SONG: ETA "
# strip: 앞뒤 공백 제거, replace: 앞의 것을 뒤의 것으로 변경
cleaned_data = raw_data.strip().replace(" ", "")
parts = cleaned_data.split(',') # ["IDOL:NewJeans", "SONG:ETA"]

result_dict = {}
for part in parts:
    key, value = part.split(':')
    result_dict[key.lower()] = value # { idol: NewJeans, song: ETA }

print(result_dict['song']) # ETA

print("======================")

# 3번, 문자열, split 조심!!
game_logs = [
    "Yujin, WIN, LOSE, WIN, WIN",
    "Karina, LOSE, WIN, LOSE, WIN",
    "Chaewon, WIN, WIN, WIN, LOSE"
]

honor_students = []
for log in game_logs:
    # parts = log.split(',')
    parts = log.strip().replace(" ", "").split(',')
    print(parts)
    player_name = parts[0]
    win_count = parts.count('WIN')
    print(win_count)

    if win_count >= 2:
        honor_students.append(player_name)

print(honor_students)

print("======================")

# 4번, 리스트 컴프리헨션, find
# find는 발견이 안 됐을때 -1
file_list = ["script.py", "image.jpg", "main.py", "style.css", "utils.py"]
py_files = [ name.replace(".py", "") for name in file_list if name.find(".py") != -1 ]
print(py_files)
print(len(py_files))