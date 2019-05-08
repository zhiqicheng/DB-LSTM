import os
path = '/DATACENTER1/hq/jpegs_256/'
with open('train_rgb_split1.txt', 'r') as f:
	with open('new_train_rgb.txt', 'w+') as ff:
		for line in f:
			line = line.strip().split(' ')
			l = line[0].split('/')
			ff.writelines(path+l[6]+' '+line[1]+' '+line[2]+'\n')
			