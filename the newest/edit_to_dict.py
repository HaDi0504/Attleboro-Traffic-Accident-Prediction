import time
import pandas as pd
from datetime import datetime
import numpy as np
import json
excel_data = pd.read_excel("center.xls").iloc[6:48,3:]

excel_data_list = np.array(excel_data)
#print (excel_data_list)

key_num = 1

dict_data = {}
print (1,excel_data_list.shape)
for i in range(1,excel_data_list.shape[0]):
	for j in range(1,excel_data_list.shape[1]):
		postion = (excel_data_list[i][0],excel_data_list[0][j])
		dict_data[str(key_num)]=postion
		key_num = key_num+1
		print (postion)
	

print (dict_data)
with open("dict.txt","w") as f:
	json.dump(dict_data,f,indent=4)
