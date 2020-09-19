# -*- coding: utf-8 -*-

import json

with open('society.json', 'r', encoding='utf-8') as reader, open("data\society.json", "w", encoding='utf-8') as writer:
	datas = json.load(reader)
	for x in datas:
		line = json.dumps(x, ensure_ascii=False)
		writer.writelines(f"{line}\n")