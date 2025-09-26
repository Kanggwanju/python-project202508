
my_playlist = ['Ditto', 'ETA', 'Hype Boy']
print(my_playlist)

# 리스트 요소 수정
my_playlist[1] = 'Super Shy'
print(my_playlist)

# 메서드
# append, 리스트에 값 추가 (맨 끝)
my_playlist.append('Attention')
print(my_playlist)

# insert, 리스트에 값 추가 (중간에 끼워넣기)
my_playlist.insert(1, 'ETA')
print(my_playlist)

print('===========')

# pop, 리스트에서 값 제거 (끝 값 제거)
removed = my_playlist.pop()
print(my_playlist)
print(f'제거된 곡명: {removed}')

# pop, 리스트에서 특정 인덱스 값 제거
removed = my_playlist.pop(2)
print(my_playlist)
print(f'제거된 곡명: {removed}')

# remove 리스트에서 특정 값 지정 제거
# 내 생각: index로 찾고, 있으면 그 값 제거
removedIndex = my_playlist.index('Ditto')
my_playlist.remove('Ditto') # 제거하려는 값이 리스트에 없으면 에러
print(removedIndex, my_playlist)
