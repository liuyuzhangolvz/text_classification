# -*- coding: utf-8 -*-

import json
import os

path_raw = 'data_raw'
path_final = 'data'

dirs = os.listdir(path_raw)

for dir in dirs:
	new_path = os.path.join(path_raw, dir)
	os.makedirs(dir[:-5], exist_ok=True)
	with open(new_path, 'r', encoding='utf-8') as reader:
		datas = json.load(reader)
		nums = 0
		for x in datas:
			nums += 1
			content_faname = os.path.join(dir[:-5], str(nums) + '.txt')
			#os.makedirs(content_faname, exist_ok=True)
			line = json.dumps(x['content'], ensure_ascii=False)
			with open(content_faname, "w", encoding='utf-8') as writer:
				writer.writelines(f"{line}\n")