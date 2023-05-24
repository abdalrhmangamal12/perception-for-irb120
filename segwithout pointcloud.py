import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pixellib
from pixellib.instance import instance_segmentation
import pyrealsense2 as rs
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

segment_frame = instance_segmentation(infer_speed = "fast")
segment_frame.load_model("/home/abdo_gamal/gp_ws/src/semantic_segmentation_ros/instance_seg/mask_rcnn_coco.h5")
target_classes = segment_frame.select_target_classes(person =True)



# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)

if __name__ == "__main__":

    
    segment_frame = instance_segmentation(infer_speed = "fast")
    segment_frame.load_model("mask_rcnn_coco.h5")

    while True:
                # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue    

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        image= np.asanyarray(color_frame.get_data())
      #  d_image=np.asanyarray(depth_frame.get_data())

        """ Prediction """
   
    #    color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segmask,output=segment_frame.segmentFrame(image,segment_target_classes=target_classes,  show_bboxes=True)

        masked_image=output
  #      print( segmask['rois'])
        
     #   print(segmask['masks'].shape)
     #   masked_img=segmask['masks'][:,:,0]
    #    for i in range((segmask['masks'].shape[2]-1)):
           
    #       masked_img=np.logical_or(masked_img,segmask['masks'][:,:,i+1])
    #    masked_img = masked_img.astype(np.uint8) *255

  #      print(masked_img.shape)
      #  print(segmask.keys())
  #      if segmask['masks'].shape[2]==1:
  #         cv2.imshow("masked_results",np.squeeze(segmask['masks']))
        cv2.imshow("masked_results",masked_image)
      #  cv2.imshow("masked_",masked_img)
            # cv2.imshow("Recognition result depth",depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
        
