import numpy as np
from PIL import Image, ImageDraw

def getNorm(e):
    """替代cv2.normalize，实现0-255范围归一化"""
    if e is None:
        print("utils.py getNorm Error #001")
        return None
    e = e.copy()
    e[e < 0.0] = 0.0  # 移除负值
    
    # 计算min-max归一化
    min_val = e.min()
    max_val = e.max()
    
    if max_val - min_val < 1e-6:  # 避免除零
        normalizedImg = np.zeros_like(e, dtype=np.float32)
    else:
        normalizedImg = (e - min_val) / (max_val - min_val) * 255.0
    
    return normalizedImg.astype(np.float32)


def detect_blobs(image, min_cutoff=20, blob_threshold=220, verbose=False):
    """替代cv2.findContours实现 blob 检测"""
    if image is None:
        print("detect_blobs() Error #001")
        return None

    lstBlob = []
    lstMin = []
    lstMax = []
    lstXY = []
    
    image = image.clip(0, 255)
    
    if np.any(image > min_cutoff):
        # 归一化到0-255并转为uint8
        normalized = getNorm(image).clip(0, 255).astype(np.uint8)
        # 边缘填充（替代cv2.copyMakeBorder）
        large = np.pad(normalized, 1, mode='constant')
        # 反转图像（替代cv2.bitwise_not）
        inverted = 255 - large
        # 阈值处理（替代cv2.threshold）
        thresh = (inverted < blob_threshold).astype(np.uint8) * 255

        # 转换为PIL图像进行轮廓检测
        pil_img = Image.fromarray(thresh)
        # 寻找轮廓（使用PIL的find_contours替代cv2.findContours）
        # 注意：PIL的find_contours返回值格式与cv2不同，需要转换
        contours = []
        for contour in pil_img.find_contours():
            # 转换为numpy数组并调整形状
            cnt = np.array(contour).reshape(-1, 1, 2).astype(np.int32)
            area = cv2_contour_area(cnt)  # 使用自定义面积计算
            if 2 < area < (large.shape[0]-1)*(large.shape[1]-1):
                contours.append((cnt, area))

        # 按面积排序（保持原逻辑）
        contours.sort(key=lambda x: x[1])
        contours = [c[0] for c in contours]

        for max_contour in contours:
            # 计算边界框（替代np.max/np.min）
            points = max_contour.reshape(-1, 2)
            xmax, ymax = points.max(axis=0)
            xmin, ymin = points.min(axis=0)
            # 边界处理
            xmax = min(xmax + 1, large.shape[1])
            ymax = min(ymax + 1, large.shape[0])
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            # 提取blob
            blob = large[ymin:ymax, xmin:xmax].astype(np.float32)
            
            # 计算质心（替代cv2.moments）
            m = contour_moments(max_contour)
            cX = int(m["m10"] / m["m00"]) if m["m00"] != 0 else 0
            cY = int(m["m01"] / m["m00"]) if m["m00"] != 0 else 0
            
            lstBlob.append(blob)
            lstMin.append(xmax - xmin)
            lstMax.append(ymax - ymin)
            lstXY.append([cX, cY])
        
        return lstBlob, lstMin, lstMax, lstXY
    else:
        if verbose:
            print(f"detect_blobs() WARNING: image values are smaller than cutoff. cutoff value: {min_cutoff:.01f}")
        return [], [], [], []


def cv2_contour_area(contour):
    """替代cv2.contourArea计算轮廓面积"""
    # 使用Shoelace公式
    points = contour.reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def contour_moments(contour):
    """替代cv2.moments计算轮廓矩"""
    points = contour.reshape(-1, 2).astype(np.float32)
    x, y = points[:, 0], points[:, 1]
    m00 = len(points)  # 简化版面积矩（实际应使用积分，这里近似）
    m10 = np.sum(x)
    m01 = np.sum(y)
    return {"m00": m00, "m10": m10, "m01": m01}


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    if pad_width[1] != 0:
        vector[-pad_width[1]:] = pad_value


def getBlobBasedOnXYSize(matrix, xy, size):
    xy = np.round(xy).astype(int)
    size = np.ceil(size / 2).astype(int)
    xMax = xy[0] + size[1] + 1
    yMax = xy[1] + size[0] + 1
    xMin = xy[0] - size[1] - 1
    yMin = xy[1] - size[0] - 1
    
    lowerY, upperY = max(yMin, 0), min(yMax, matrix.shape[0])
    lowerX, upperX = max(xMin, 0), min(xMax, matrix.shape[1])
     
    blob = matrix[lowerY:upperY, lowerX:upperX]
    
    template = np.zeros([(size[0]+1)*2, (size[1]+1)*2])
    
    try:
        template[0:(size[0]+1)*2, 0:(size[1]+1)*2] = blob
    except ValueError:
        y_lower_boundary = np.abs(min(yMin, 0))
        y_upper_boundary = (size[1]+1)*2 + min(0, matrix.shape[0]-yMax)
        x_lower_boundary = np.abs(min(xMin, 0))
        x_upper_boundary = (size[0]+1)*2 + min(0, matrix.shape[1]-xMax)

        template[y_lower_boundary:y_upper_boundary, x_lower_boundary:x_upper_boundary] = blob

    return template.astype(np.float32)


def aligneCentre(AprilTagPixelSizeInMM, PixelSizeInMM, markerPixels, img):
    """注意：原函数中的aligneImageECC没有cv2替代方案，这里保留框架并提示"""
    shape = img.shape
    size = int(np.round(AprilTagPixelSizeInMM / PixelSizeInMM * markerPixels))
    clear = np.zeros(shape, np.float32)
    xy = np.round(np.array(shape)/2).astype(int)
    dim = np.array([xy - size/2, xy + size/2]).astype(int)
    # 确保索引有效
    dim[0, 0] = max(0, dim[0, 0])
    dim[1, 0] = min(shape[0], dim[1, 0])
    dim[0, 1] = max(0, dim[0, 1])
    dim[1, 1] = min(shape[1], dim[1, 1])
    
    clear[dim[0,0]: dim[1,0], dim[0,1]: dim[1,1]] = np.ones((
        dim[1,0]-dim[0,0], 
        dim[1,1]-dim[0,1]
    ), np.float32) * img.mean()
    
    # 注意：ECC图像对齐是cv2特有功能，无直接替代方案
    # 这里返回原始图像作为替代，实际应用中可能需要实现简单对齐逻辑
    print("警告：aligneImageECC无cv2替代方案，返回原始图像")
    return img  # 原逻辑中的imgA = aligneImageECC(clear, img)