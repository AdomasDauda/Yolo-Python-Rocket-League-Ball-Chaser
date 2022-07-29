import os
from PIL import Image


folder = 'D:\RocketLeagueAi\dataset3'

for i in os.walk(folder):
    for j in range(len(i[2])):
        if '.png' in i[2][j]:
            filename = i[2][j]
            print(i[2][j])
            img_png = Image.open(f'{folder}\{filename}')
            # converting to jpg file
            #saving the jpg file
            filename = filename.replace('.png','')
            img_png.save(f'{folder}\{filename}.jpg')
            os.remove(f'{folder}\{filename}.png')


list = []
for i in os.walk(folder):
    for j in range(len(i[2])):
        filename = i[2][j]
        filename = filename.replace('.jpg','')
        filename = filename.replace('.txt','')
        list.append(filename)

print(len(list)/2)

duplicates = [name for name in list if list.count(name) > 1]

for i in os.walk(folder):
    for j in range(len(i[2])):
        filename = i[2][j]
        filename = filename.replace('.jpg','')
        filename = filename.replace('.txt','')

        if filename not in duplicates:
            print(f'removing: {filename}')
            os.remove(f'{folder}\{filename}.jpg')
