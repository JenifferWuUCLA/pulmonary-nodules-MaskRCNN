import numpy as np
from xlwt import *

enquiry_file = './data/weibo_post'
answer_file = './data/weibo_response'

len_matrix = np.zeros((20, 20), dtype='int64')

f_enq = open(enquiry_file, 'r')
f_ans = open(answer_file, 'r')
for le in f_enq:
    la = f_ans.readline()

    la = la.strip().split(' ')
    le = le.strip().split(' ')

    len_matrix[int(len(le) / 5)][int(len(la) / 5)] += 1
f_enq.close()
f_ans.close()

w = Workbook()
ws = w.add_sheet('sheet')
for i in range(20):
    for j in range(20):
        ws.write(i, j, len_matrix[i][j])
w.save('enq_ans_length_matrix.xls')
print "Saved to enq_ans_length_matrix.xls. colume: length of enquiries, row: length of answers"
