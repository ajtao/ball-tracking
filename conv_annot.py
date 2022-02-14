# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 11:17:08 2022

@author: steve
"""
import argparse
import pandas as pd
import os
import cv2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_dir", help="Directory with video and text annotations")
    args = parser.parse_args()

    for filename in os.listdir(args.raw_data_dir):

        pre, ext = os.path.splitext(filename)
        if ext == '.txt':
            txt_path = os.path.join(args.raw_data_dir, pre + ext)
            video_path = os.path.join(args.raw_data_dir, pre + '.mp4')
            annot_path = os.path.join(args.raw_data_dir, pre + '.csv')

            # open the video file to determine frame size
            cap = cv2.VideoCapture(video_path)
            frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            df = pd.read_csv(txt_path,
                             header=None,
                             names=['Frame',
                                    'Ball',
                                    'bb_left',
                                    'bb_top',
                                    'bb_width',
                                    'bb_height',
                                    'conf',
                                    'wc_x',
                                    'wc_y',
                                    'wc_z'])

            df['x'] = (df['bb_left'] + df['bb_width']/2) / frame_width
            df['y'] = (df['bb_top'] + df['bb_height']/2) / frame_height

            # reindex to fill in missing frames
            full_index = pd.Index(range(0, frame_count), name='Frame')
            df = df.set_index('Frame').reindex(full_index).reset_index()
            df.loc[df['Ball'].isna(), ['Ball', 'x', 'y']] = [0, -1, -1]
            df['Ball'] = df['Ball'].astype('int')

            df[['Frame', 'Ball', 'x', 'y']].to_csv(annot_path, index=False)
