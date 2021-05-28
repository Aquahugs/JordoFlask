import os 
import shutil

link = 'images/raw'
files = os.listdir(link)
folders = [ file for file in files if os.path.isdir(link + '/' + file) ]

for folder in folders:
    print(folder)

    # try:
    shutil.move(link + '/' + folder, 'images')

    cur = 'images/' + folder
    dirs = os.listdir(cur)
    found = False
    while len(dirs) > 0 and not found:
        for file in dirs:
            if os.path.isdir(cur + '/' + file):
                cur += '/' + file
                dirs = os.listdir(cur)
                break

            if os.path.isfile(cur + '/' + file):
                if '.' not in file:
                    old_image = cur + '/' + file 
                    found = True
                    break
                else:
                    name, ext = os.path.splitext(dirs)
                    if ext.lower() in ['jpg', 'png', 'jpeg', 'bmp']:
                        old_image = cur + '/' + file 
                        found = True
                        break


    new_image = link + '/' + folder
    os.rename(old_image, new_image)
    shutil.rmtree('images/' + folder)
  # except:
  #   print('Error')
  #   pass