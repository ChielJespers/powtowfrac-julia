import os

for i in range(300):
    inv_i = 600 - i
    os.system('convert -flip output2/frame{:05d}.png output2/frame{:05d}.png'.format(i, inv_i))