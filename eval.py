#%%
import numpy as np
import cv2
import os 
import math

def iou_sum(total_area_intersect, total_area_union):
  # Make sure the segmentation maps have the same shape
  return total_area_intersect / total_area_union

def cal_single_intersect_union(seg1, seg2):
  # Make sure the segmentation maps have the same shape
  assert seg1.shape == seg2.shape

  seg1 = seg1.astype(np.int64)
  seg2 = seg2.astype(np.int64)

  # Calculate the intersection of the two segmentation maps
  intersection = (seg1 & seg2).sum()

  # Calculate the union of the two segmentation maps
  union = (seg1 | seg2).sum()
  return intersection, union



def iou_gingle(seg1, seg2):
  # Make sure the segmentation maps have the same shape
  assert seg1.shape == seg2.shape

  seg1 = seg1.astype(np.int64)
  seg2 = seg2.astype(np.int64)

  # Calculate the intersection of the two segmentation maps
  intersection = (seg1 & seg2).sum()

  # Calculate the union of the two segmentation maps
  union = (seg1 | seg2).sum()

  # Return the IoU as the intersection over union
  return intersection / union

def mse(gt, pred):
  # Make sure the segmentation maps have the same shape
  assert gt.shape == pred.shape

  # Calculate the intersection of the two segmentation maps
  intersection = (gt & pred).sum()

  # Calculate the union of the two segmentation maps
  union = (gt | pred).sum()

  pred_sum = pred.sum()
  gt_sum = gt.sum()

  # single_error = np.sqrt((pred_sum - intersection + gt_sum - intersection) / (gt.shape[0] * gt.shape[1]))
  single_error = (pred_sum - intersection + gt_sum - intersection) / (gt.shape[0] * gt.shape[1])


  # Return the IoU as the intersection over union
  return single_error

def Mse_prob(gt, pred):
  error_sum = 0
  for i in range(gt.shape[0]):
    for j in range(gt.shape[1]):
      pixel_error = (gt[i][j] - pred[i][j]) ** 2
      error_sum += pixel_error
  Mse = error_sum / (gt.shape[0] * gt.shape[1])
  return Mse

def cal_tp_fp_fn(seg1, seg2):
  # Make sure the segmentation maps have the same shape
  assert seg1.shape == seg2.shape
  seg1 = seg1.astype(np.int64)
  seg2 = seg2.astype(np.int64)
  # Calculate the true positives (tp), false positives (fp), and false negatives (fn)
  tp = (seg1 & seg2).sum()
  fp = (seg1 & ~seg2).sum()
  fn = (~seg1 & seg2).sum()

  return tp, fp, fn

def f1_score(tp, fp, fn):
  # Calculate the precision, recall, and F1-score
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f1 = 2 * precision * recall / (precision + recall)

  # Return the F1-score
  return f1

def f_beta_score_single(seg1, seg2, beta):
      # Make sure the segmentation maps have the same shape
  assert seg1.shape == seg2.shape
  seg1 = seg1.astype(np.int64)
  seg2 = seg2.astype(np.int64)

  # Calculate the true positives (tp), false positives (fp), and false negatives (fn)
  tp = (seg1 & seg2).sum()
  fp = (seg1 & ~seg2).sum()
  fn = (~seg1 & seg2).sum()

  # Calculate the precision, recall, and F1-score
  precision = tp / max(1, (tp + fp))
  recall = tp / max(1,(tp + fn))
  # f_beta = (1 + beta ** 2) * precision * recall / max((beta ** 2 * precision + recall), 0.001) # 防止除0这里max 0.001 参考kerals
  f_beta = (1 + 0.3) * precision * recall / max((0.3 * precision + recall), 0.001) # 防止除0这里max 0.001 参考kerals

  # Return the F1-score
  return f_beta


# Import NumPy
# Create two random segmentation maps
pred_root_path = 'sample_SMOKE1K'
# pred_root_path = '/workspace/ylj/Transmission-BVM/results_final_pred_out1/SMOKE5K/sal'
gt_root_path = 'data_dir_smoke/SMOKE1K/test/gt'

gt_list = os.listdir(gt_root_path)
pred_list = os.listdir(pred_root_path)

iou_sum = 0.
mse_sum = 0.
tp_sum = 0.
fp_sum = 0.
fn_sum = 0.
f1_sum = 0.
f2_sum = 0.
num_nan = 0

intersection = 0
union = 0



for pred_map in pred_list:
    if '.jpg' not in pred_map:
      continue
    pred_map_path = os.path.join(pred_root_path, pred_map)
    gt_map_path = os.path.join(gt_root_path, pred_map).replace('.jpg', '.png').replace('_output_ens', '')
    # pred_map_path = '/workspace/ylj/Transmission-BVM/pred/1499546443_+00180.png'
    # gt_map_path = '/workspace/ylj/Transmission-BVM/dataset/SMOKE5K_docker/SMOKE5K_origin/test/gt_/1499546443_+00180.png'
    pred = cv2.imread(pred_map_path, -1)
    gt = cv2.imread(gt_map_path, -1)
    pred = cv2.resize(pred, [256, 256])
    gt = cv2.resize(gt, [256, 256])
    if len(gt.shape) > 2:
        gt = gt[:, :, 2]
    if len(pred.shape) > 2:
        pred = pred[:, :, 2]
    # gt[gt > 0] = 1 # 将大于0的像素设定为1
    
    gt = gt / 128 # 将像素映射到0-255，用于计算mse
    # gt[gt > 0.5] = 1
    # gt[gt <= 0.5] = 0
    # if gt.shape != pred.shape:
    #     pred = cv2.resize(pred, list(reversed(gt.shape)))
    # pred[pred > 0] = 1
    pred = pred / 255 # 将像素映射到0-255，用于计算mse

    
    
    single_mse = np.mean((gt - pred)**2)
    mse_sum += single_mse

    # 卡阈值
    # threshold = 2 * pred.mean() # SOD中的阈值选取方法
    threshold = 0.1
    
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0

    intersection_single, union_single = cal_single_intersect_union(gt, pred)
    intersection += intersection_single
    union += union_single

    iou_sum += iou_gingle(gt, pred)
    tp, fp, fn = cal_tp_fp_fn(gt, pred)
    tp_sum += tp
    fp_sum += fp
    fn_sum += fn
    # if single_mse > 0.01:
    #   print(single_mse)
    #   print(pred_map_path)
    #   print(gt_map_path)

    f1_single = f_beta_score_single(pred, gt, beta=1)
    f1_sum += f1_single
    if math.isnan(f1_single):
      print(f1_single)
      print(pred_map_path)
      print(gt_map_path)


mIoU = iou_sum / len(pred_list)
mMse = mse_sum / len(pred_list)
f1 = f1_score(tp_sum, fp_sum, fn_sum)
f1_by_single = f1_sum / (len(pred_list) - num_nan)
IoU_total = intersection / union
print('IoU_total: ' + str(IoU_total))
print('mIoU: ' + str(mIoU))
print('mMse: ' + str(mMse))
print('f1: ' + str(f1))
print('f1_by_single: ' + str(f1_by_single))
