import os
import numpy as np 

def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

def cal_dist(pose1, pose2):
    dist = 0.0
    cnt=0
    for i in range(pose1.shape[0]):
        p1_x = pose1[i][0]
        if p1_x < 0:
            continue
        p1_y = pose1[i][1]
        p2_x = pose2[i][0]
        if p2_x < 0:
            continue
        p2_y = pose2[i][1]
        cnt+=1
        dist += np.sqrt((p1_x - p2_x) **2 + (p1_y - p2_y) **2)
    return dist / cnt

def DTWDistance_pose(seq_s, seq_t):
    DTW={}
    import pdb
    
    for i in range(seq_s.shape[0]):
        DTW[(i, -1)] = float('inf')
    for i in range(seq_t.shape[0]):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0
    

    for i in range(seq_s.shape[0]):
        for j in range(seq_t.shape[0]):
            dist = cal_dist(seq_s[i], seq_t[j])
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
    
    return np.sqrt(DTW[(seq_s.shape[0]-1, seq_t.shape[0]-1)])



def kneecurl_dtw(seq_student, seq_teacher):
   #from posedet import get_seq
   #seq_student = get_seq('test1.mp4')
   #seq_teacher = get_seq('template1.mp4')
   #import pdb
   #pdb.set_trace()
   #seq_student = np.load('seq_s.npy')
   #seq_teacher = np.load('seq_t.npy')
   dis = DTWDistance_pose(seq_student, seq_teacher)

   return dis

# if __name__ == '__main__':
#    main()