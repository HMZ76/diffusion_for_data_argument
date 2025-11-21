from PIL import Image
import numpy as np
import os.path

markerDict = {}

def readMarker(markerId, markerType, markerResolution=None, debug=False, path="./"):
    if markerResolution is None:
        key = f"{markerType}_{markerId}"
        if key in markerDict:
            return markerDict[key]        
        else:
            # 构建文件路径
            if markerType in ["36h11", "16h5"]:
                x1, x2 = markerType.split("h")
                file = f"{path}tag{x1}h{x2}/tag{x1}_%02d_%05d.png" % (int(x2), int(markerId))
            elif markerType == 'shape':
                file = f"{path}{markerType}/{markerId}_8mm.png"
            else:
                file = f"{path}{markerId}.png"
            
            if debug:
                print(file)
            
            if os.path.isfile(file):
                # 用PIL读取图像并转换为numpy数组（处理为单通道）
                with Image.open(file) as img:
                    # 转换为灰度图并转为数组
                    marker_np = np.array(img.convert('L'), dtype=np.float32)
                    # 反转颜色并归一化到[0,1]（对应原cv2的(255 - ...)/255.0）
                    marker = (255.0 - marker_np) / 255.0
                
                markerDict[key] = marker
                return marker
            else:
                print(f"File {file} does not exist")
                return None
    else:
        key = f"{markerType}_{markerId}_{markerResolution}"
        if key in markerDict:
            return markerDict[key]
        else:
            # 构建文件路径
            if markerType in ["36h11", "16h5"]:
                x1, x2 = markerType.split("h")
                file = f"{path}tag{x1}h{x2}/tag{x1}_%02d_%05d.png" % (int(x2), int(markerId))
            else:
                file = f"{path}{markerType}/{markerId}_8mm.png"
            
            if debug:
                print(file)
            
            if os.path.isfile(file):
                with Image.open(file) as img:
                    marker_np = np.array(img.convert('L'), dtype=np.float32)
                    marker = (255.0 - marker_np) / 255.0
            else:
                print(f"File {file} does not exist")
                return None

            # 创建目标尺寸数组并居中放置marker
            marker2 = np.zeros((markerResolution, markerResolution), dtype=np.float32)
            c = np.array([markerResolution, markerResolution]) // 2
            wh = np.array(marker.shape) // 2
            x1, y1 = c - wh
            x2, y2 = c + wh
            marker2[x1:x2, y1:y2] = marker

            markerDict[key] = marker2
            return marker2

def load_tag(size, marker_type, markerId, angle, PixelSizeInMM=4.022, MARKER_PATH='./'):
    AprilTagPixelSizeInMM = size
    # 读取原始marker
    template = readMarker(markerId, marker_type, markerResolution=None, path=f"{MARKER_PATH}")
    if template is None:
        return None

    # 根据marker类型进行缩放
    if marker_type in ['16h5', '36h11']:
        # 计算缩放比例并缩放
        s_ratio = (PixelSizeInMM / AprilTagPixelSizeInMM)
        s = np.round(np.array(template.shape) / s_ratio * 4).astype(int)
        # 转换为PIL图像进行缩放（ nearest邻插值）
        pil_img = Image.fromarray((template * 255).astype(np.uint8))  # 转回0-255范围
        resized_img = pil_img.resize(tuple(s[::-1]), resample=Image.NEAREST)  # PIL尺寸是(w,h)，数组是(h,w)
        template = np.array(resized_img, dtype=np.float32) / 255.0  # 转回0-1范围
    else:
        s_ratio = size / PixelSizeInMM
        s = np.round(s_ratio * 4).astype(int)
        # 转换为PIL图像进行缩放（ Lanczos插值）
        pil_img = Image.fromarray((template * 255).astype(np.uint8))
        resized_img = pil_img.resize((s, s), resample=Image.LANCZOS)
        template = np.array(resized_img, dtype=np.float32) / 255.0
        template = np.clip(template, 0, 1.0)  # 原逻辑是0-255，这里对应0-1

    # 旋转图像（替代imutils.rotate_bound，保持边界完整）
    # 转换为PIL图像旋转
    pil_img = Image.fromarray((template * 255).astype(np.uint8))
    rotated_img = pil_img.rotate(360-angle, expand=True, resample=Image.BILINEAR)  # expand=True保持完整
    rotated_np = np.array(rotated_img, dtype=np.float32) / 255.0  # 转回0-1范围

    # 归一化（假设process.getNorm是简单归一化到0-1，若有其他逻辑需调整）
    # 这里简化为保持0-1范围（因前面已处理），如需标准化可补充
    templateRet = rotated_np.astype(np.float32)

    # 调整尺寸到128x128（居中裁剪或填充）
    target_size = 128
    temp = np.array([target_size, target_size])
    current_h, current_w = templateRet.shape
    w_diff = target_size - current_w
    h_diff = target_size - current_h

    if w_diff < 0 or h_diff < 0:
        # 需要裁剪（居中裁剪到目标尺寸）
        c = np.array([current_h, current_w]) // 2
        wh = temp // 2
        x1, y1 = c - wh
        x2, y2 = c + wh
        # 处理边界情况（避免索引越界）
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(current_h, x2)
        y2 = min(current_w, y2)
        templateRet = templateRet[x1:x2, y1:y2]
    elif w_diff > 0 or h_diff > 0:
        # 需要填充（居中填充）
        pad_h = (int(np.ceil(h_diff / 2)), int(h_diff - np.ceil(h_diff / 2)))
        pad_w = (int(np.ceil(w_diff / 2)), int(w_diff - np.ceil(w_diff / 2)))
        templateRet = np.pad(templateRet, (pad_h, pad_w), mode='constant')

    return templateRet