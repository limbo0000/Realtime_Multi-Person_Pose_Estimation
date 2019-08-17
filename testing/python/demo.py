import sys
import numpy as np
import cv2 
from posedet import PoseDet, get_seq, save_video
from evaluation_dtw import DTWDistance_pose
import logging 


def pre_process(video_name, save_name):
    print('processing.......')
    pose_seq, frames = get_seq(video_name, rate=5, display=True)
    save_video_name = save_name+'.avi'
    save_pose_seq_name = save_name+'.npy'
    fwidth = frames[0].shape[1]
    fheight = frames[0].shape[0]
    save_video(save_video_name, frames, fwidth, fheight, fps=10)
    np.save(save_pose_seq_name, pose_seq)
    return pose_seq

def main():
    
    student_name = sys.argv[1]
    s_save_name = sys.argv[2]

    seq_s = pre_process(student_name, s_save_name)
    if len(sys.argv) > 3:
        teacher_name = sys.argv[3]
        if len(sys.argv) ==4:
            seq_t = np.load(teacher_name)
        else: 
            t_save_name = sys.argv[4]
            seq_t = pre_process(teacher_name, t_save_name)

        dist = DTWDistance_pose(seq_s, seq_t)
        print(dist)


if __name__ == "__main__":
    main()