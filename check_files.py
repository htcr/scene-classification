import os
andrew_id = 'chetang'


if ( os.path.isfile('../'+andrew_id+'/code/visual_words.py') and \
os.path.isfile('../'+andrew_id+'/code/visual_recog.py') and \
os.path.isfile('../'+andrew_id+'/code/network_layers.py') and \
os.path.isfile('../'+andrew_id+'/code/deep_Recog.py') and \
os.path.isfile('../'+andrew_id+'/code/util.py') and \
os.path.isfile('../'+andrew_id+'/code/main.py') and \
os.path.isfile('../'+andrew_id+'/'+andrew_id+'_hw1.pdf') ):
    print('file check passed!')
else:
    print('file check failed!')

#modify file name according to final naming policy
#images should be included in the report
