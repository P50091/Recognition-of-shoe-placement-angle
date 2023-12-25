import cv2
import numpy as np
import os
import shutil

def process_images(input_folder, input1_folder,output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有文件
    filenames = os.listdir(input_folder)

    #计数器
    count =0
    # 遍历每个文件
    for filename in filenames:
        # 读取图像
        image = cv2.imread(os.path.join(input_folder, filename))
        image1 = cv2.imread(os.path.join(input1_folder, filename))
        # 处理图像
        processed_image = process_image(image,image1)
        # 保存结果到输出文件夹
        cv2.imwrite(os.path.join(output_folder, filename), processed_image)
        count+=1
        print(f"已处理 {count} / {len(filenames)} 张图片，{filename} 保存成功")

    shutil.rmtree(input_folder)
def select_largest_contour(contours):
    # 计算每个轮廓的长度
    lengths = [len(contour) for contour in contours]
    # 找到长度最大的轮廓的索引
    largest_contour_index = np.argmax(lengths)
    # 返回长度最大的轮廓
    return contours[largest_contour_index]

def normalize(lst):
    if not lst:
        raise ValueError("列表不能为空!")
    arr = np.array(lst)
    return (arr - arr.min()) / (arr.max() - arr.min())

def find_longest_line(w1,w2,lines):

    # 计算每条直线的垂直位置（y坐标的最大值）
    positions = [max(line[0][1], line[1][1]) for line in lines]
    new_positions = normalize([float(x) for x in positions])

    # 计算每条直线的长度
    lengths = [np.sqrt((line[0][0] - line[1][0]) ** 2 + (line[0][1] - line[1][1]) ** 2) for line in lines]
    lengths = normalize(lengths)

    # 计算每条直线的加权值并添加到结果列表中
    result = [w1 * pos + w2 * length for pos, length in zip(new_positions, lengths)]
    longest_line_index = np.argmax(result)
    line = lines[longest_line_index]

    return line

def process_image(image,image1):
    w1 =0.7
    w2 = 0.3
    # 将图像转换为灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行边缘检测
    edges = cv2.Canny(gray, 50, 150)
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #找最大的contour
    contour = select_largest_contour(contours)
    # 计算最小外接多边形
    hull = cv2.convexHull(contour)
    # 在原始图像上绘制多边形
    cv2.polylines(image, [hull], True, (0, 0, 255), 3)

    # 计算最小外接三角形
    area, triangle = cv2.minEnclosingTriangle(contour)
    # 将三角形的顶点从浮点数转换为整数
    triangle = np.intp(triangle)
    # 按照x值对顶点进行排序
    sorted_indices = np.argsort(triangle[:, 0, 1])
    triangle = triangle[sorted_indices]
    # 找到与第一个点的y值最接近的点
    second_point_index = np.argmin(np.abs(triangle[1:, 0, 0] - triangle[0, 0, 0])) + 1
    # 将第二个点移动到第一个点之后
    triangle[1, :], triangle[second_point_index, :] = triangle[second_point_index, :], triangle[1, :].copy()
    # 在原始图像上绘制三角形
    # cv2.polylines(image, [triangle], True, (0, 255, 0), 2)

    lines = []
    # 遍历所有点
    for i in range(len(hull)):
        # 获取当前点和下一个点
        p1 = tuple(hull[i][0])
        p2 = tuple(hull[(i + 1) % len(hull)][0])
        # 计算斜率
        if p2[0] - p1[0] == 0:  # 防止除数为0
            slope = float('inf')
        else:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        # 计算与水平线的夹角
        angle = np.abs(np.arctan(slope)) * 180 / np.pi
        # 如果夹角小于或等于80度，则将两点之间的直线添加到列表中
        if angle < 40:
            lines.append((p1, p2))

    if lines:
        line = find_longest_line(w1,w2,lines)
        cv2.line(image, line[0], line[1], (0, 255, 0), 3)

        p1 = line[0]
        p2 = line[1]
        dy = p1[1] - p2[1]  # 取反，使y轴正方向向上
        dx = p2[0] - p1[0]
        # 使用arctan2计算角度
        angle = np.arctan2(dy, dx) * 180 / np.pi
        # 将角度转换为0-180度的范围
        if angle < 0:
            angle += 180
        if angle <5:
            angle=0

        if angle >175:
            angle =180

        if angle ==180:
            if triangle[1][0][0] > triangle[2][0][0]:
                angle =0
        # cv2.putText(image, f"Angle: {int(angle)}", (70,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),2, cv2.LINE_AA)
        cv2.putText(image1, f"Angle: {int(angle)}", (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

    return image1

if __name__ == '__main__':
    process_images("mask_images", 'data/images',"output")