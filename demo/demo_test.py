from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2
import sys

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image_arr = ['demo_test_img1.png', 'demo_test_img2.jpg']
if (len(sys.argv) != 2):
    print('invalid # of arguments')
else:
    image = cv2.imread(image_arr[int(sys.argv[1])])
predictions = coco_demo.run_on_opencv_image(image)

cv2.imshow('pred', predictions)
cv2.waitKey(0)
# cv2.imwrite('output1.jpg', predictions)