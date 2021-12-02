import cv2
import numpy as np
from PIL import Image
from aip import AipOcr
import chardet

def stretch(img):
    #图像拉伸函数
    max_ = float(img.max())  #float() 函数用于将整数和字符串转换成浮点数
    min_ = float(img.min())

    for i in range(img.shape[0]):    #(img.shape[0])读入时的图片的高度height
        for j in range(img.shape[1]):   #(img.shape[1])读入时的图片的宽度weight
            img[i, j] = (255 / (max_ - min_)) * img[i, j] - (255 * min_) / (max_ - min_)  #灰度拉伸公式
    return img


def dobinaryzation(img):
    #二值化处理函数
    max = float(img.max())
    min = float(img.min())

    x = max - ((max - min) / 2)
    # 二值化,返回阈值ret  和  二值化操作后的图像thresh
    ret, threshedimg = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)   #阈值x-255之间  cv2.THRESH_BINARY表示阈值的二值化操作，大于阈值使用255表示，小于阈值使用0（黑色）表示
    # 返回二值化后的黑白图像
    return threshedimg


def find_retangle(contour):
    #寻找矩形轮廓
    y, x = [], []

    for p in contour:
        y.append(p[0][0])  #append() 方法用于在列表末尾添加新的对象。
        x.append(p[0][1])

    return [min(y), min(x), max(y), max(x)]


def locate_license(img, orgimg):
    #定位车牌号
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#contours(轮廓的检索模式)=cv2.RETR_EXTERNAL表示只检测外轮廓#
    # hierarchy(轮廓的近似办法)=cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息

    # 找出最大的三个区域
    blocks = []
    for c in contours:
        # 找出轮廓的左上点和右下点，由此计算它的面积和长宽比
        r = find_retangle(c)
        a = (r[2] - r[0]) * (r[3] - r[1])  #面积
        s = (r[2] - r[0]) / (r[3] - r[1])  #长宽比

        blocks.append([r, a, s])

    # 选出面积最大的3个区域    对blocks列表的s(即长宽比)元素进行排列
    blocks = sorted(blocks, key=lambda b: b[2])[-3:] #对元素第3个字段排序，则key=lambda y: y[2],这里y可以是任意字母，
    #当待排序列表的元素由多字段构成时，我们可以通过sorted(iterable，key，reverse)的参数key来制定我们根据那个字段对列表元素进行排序。
    # 使用颜色识别判断找出最像车牌的区域
    maxweight, maxinedx = 0, -1
    for i in range(len(blocks)):
        b = orgimg[blocks[i][0][1]:blocks[i][0][3], blocks[i][0][0]:blocks[i][0][2]]
        # RGB转HSV
        hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        # 蓝色车牌范围
        lower = np.array([100, 50, 50])
        upper = np.array([140, 255, 255])
        # 根据阈值构建掩模
        mask = cv2.inRange(hsv, lower, upper)

        # 统计权值
        w1 = 0
        for m in mask:
            w1 += m / 255

        w2 = 0
        for w in w1:
            w2 += w

        # 选出最大权值的区域
        if w2 > maxweight:
            maxindex = i
            maxweight = w2

    return blocks[maxindex][0]


def find_license(img):
    '''预处理'''
    # 压缩图像
    img = cv2.resize(img, (400, int(400 * img.shape[0] / img.shape[1])))

    # 灰度图
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('grayimg', grayimg)
    # 对灰度图像像素进行拉伸，使图片的像素值拉伸到整个像素空间，提高图像像素的对比度
    stretchedimg = stretch(grayimg)
    #cv2.imshow('stretchedimg', stretchedimg)

    # 进行开运算，用来去噪声
    # 先定义一个元素结构
    r = 16
    h = w = r * 2 + 1
    kernel = np.zeros((h, w), dtype=np.uint8)   #np.zeros((h, w)) 生成h行w列的零矩阵  想为图像创建一个容器，需要指定dtype=np.uint8
    cv2.circle(kernel, (r, r), r, 1, -1)  #在kernel绘制圆的图像  (r, r)圆的中心坐标  r圆的半径  厚度-1像素将以指定的颜色填充矩形形状
    # 开运算
    openingimg = cv2.morphologyEx(stretchedimg, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('openingimg', openingimg)
    # 获取两个图像之间的差分图  两幅图像做差  cv2.absdiff('图像1','图像2')
    # cv2.absdiff可以把两幅图的差的绝对值输出到另一幅图上面来
    # 利用这种办法可以去除图片中的大面积噪声
    strtimg = cv2.absdiff(stretchedimg, openingimg)
    #cv2.imshow('strtimg', strtimg)
    # 图像二值化
    binary_img = dobinaryzation(strtimg)
    # cv2.imshow('binary_img', binary_img)
    # 使用Canny函数做边缘检测
    cannyimg = cv2.Canny(binary_img, binary_img.shape[0], binary_img.shape[1])
    # cv2.imshow('cannyimg', cannyimg)
    ''' 消除小区域，保留大块区域，从而定位车牌'''
    # 进行闭运算
    kernel = np.ones((5, 19), np.uint8)
    closing_img = cv2.morphologyEx(cannyimg, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('closing_img', closing_img)
    # 进行开运算
    opening_img = cv2.morphologyEx(closing_img, cv2.MORPH_OPEN, kernel)

    # 再次进行开运算
    kernel = np.ones((11, 5), np.uint8)
    opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)
    # kernel = np.ones((5, 19), np.uint8)
    # closingimg1 = cv2.morphologyEx(openingimg, cv2.MORPH_CLOSE, kernel)
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_dilated = cv2.dilate(opening_img, kernel_2)
    # cv2.imshow('kernel_dilated', kernel_dilated)
    # 消除小区域，定位车牌位置
    rect = locate_license(kernel_dilated, img)
    return rect, img

""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

if __name__ == '__main__':
    # 读取图片
    orgimg = cv2.imread('C:/Users/Administrator/Desktop/bishe/images/47.jpg')
    rect, img = find_license(orgimg)

    # 框出车牌
    draw_image = cv2.rectangle(img, (rect[0]-2, rect[1]), (rect[2]+10, rect[3]+5), (0, 255, 0), 2)
    chepai = draw_image[rect[1]:rect[3]+5 , rect[0]-2 :rect[2]+10]
    chepai = cv2.resize(chepai, (400, 200))
    Vshow = chepai.copy()
    print(cv2.imwrite("C:/Users/Administrator/Desktop/bishe/01/1.jpg", Vshow))

    # 定义常量
    APP_ID = 'XX'  # 你百度帐号上的APP_ID
    API_KEY = 'XX'  # 你百度帐号上的API_KEY
    SECRET_KEY = 'XX'  # 你百度帐号上的SECRET_KEY

    # 初始化AipFace对象
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    image = get_file_content('C:/Users/Administrator/Desktop/bishe/01/1.jpg')  # 将左侧括号内替换为待识别的图片路径

    """ 调用通用文字识别（高精度版） """
    result = client.basicAccurate(image)
    print(result)  #输出{'words_result': [{'words': '蒙E18958'}], 'words_result_num': 1, 'log_id': 1466298976534476821}

#疫情高风险地区：呼伦贝尔市（蒙E）；中风险地区：上海浦东新区，上海青浦区（沪），辽宁省大连（辽B），云南瑞丽市（云N）  2021.12.01消息
    list1 = ['蒙E']
    list2 = ['沪','辽B','云N']
    result1 = result.get('words_result') #获取字典result的第一个key（'words_result'）的值，即[{'words': '蒙E18958'}]
    result2 = result1[0] #获取列表result1的第一个元素，即{'words': '蒙E18958'}
    result3 = result2.get('words')#获取字典result2的key('words')的值，即蒙E18958
    #print(result1)
    #print(result2)
    print(result3)
    result4 = result3[0:2] #即蒙E
    #print(result4)
    if result4 in list1:
        print("该车牌来自疫情高风险地区，请注意疫情防控")
    elif result4 in list2:
        print("该车牌来自疫情中风险地区，请注意疫情防控")

    # 显示带检测框的图像
    cv2.imshow('img', img)
    cv2.imshow('chepai', chepai)
    #cv2.imshow('rect', rect)
    # 显示原始图像
    # cv2.imshow('orgimg', orgimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()