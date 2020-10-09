import matplotlib.pyplot as plt
import numpy as np

# data = np.loadtxt("train_data/DNN_result.txt")
data = np.loadtxt("train_data/Loss_record.txt")
print(data.shape)
#print(data[1:4000])
# first = data[1:4000]
# plt.plot(first[:,0], first[:,1])
plt.plot( data[:,1])
#plt.plot(data[:,0], data[:,3])
# plt.savefig('train_data/DNN_result.png')
# plt.savefig('train_data/Loss_record.png')
# plt.show()

# import os
# import matplotlib.pyplot as plt
# data_path = "/home/bq1235/datasets/cubes/training"
# img_path = data_path + '/image_2'
# label_path = data_path + '/label_2'
# IDLst = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
# print(len(IDLst))
# buffer = []
# iou_data = []
# for idx in range(len(IDLst)):
#     with open(label_path + '/%s.txt' % IDLst[idx], 'r') as f:
#         for line in f:
#             line = line[:-1].split(' ')
#             for i in range(1, len(line)):
#                 line[i] = float(line[i])
#             Class = line[0]
#             IoU = line[8]  # added benchun 20200927 for better loss
#             top_left = (int(round(line[9])), int(round(line[10])))
#             bottom_right = (int(round(line[11] + line[9])), int(round(line[12] + line[10])))
#             Box_2D = [top_left, bottom_right]
#             buffer.append({
#                 'Class': Class,
#                 'Box_2D': Box_2D,
#                 'IoU': IoU,
#             })
#             iou_data.append(IoU)
# plt.hist(iou_data, bins=50)
# plt.savefig('train_data/data.png')
# plt.show()