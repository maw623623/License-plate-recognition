# License-plate-recognition
运行环境为Pycharm，python3.6版本，曾尝试把程序移植到树莓派上运行，但不知道为什么程序在树莓派上运行会显示error，错误如下：

针对程序中的语句：rect, img = find_license(orgimg)，和contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

报错如下：ValueError: too many values to unpack (expected 2)

将车牌从图片中框选出来并裁剪，保存图片，调用百度云API接口（通用文字识别高精度版）进行文字识别。

images存放的是识别的图片，并不是所有图片都能识别成功。文字高精度成功文件夹存放的是识别成功了的文件。

这个项目存在不足，对车牌的“云”字基本识别不出来。

车牌识别到疫情中高风险地区的车牌会提示"该车牌来自疫情高风险地区，请注意疫情防控"

#疫情高风险地区：呼伦贝尔市（蒙E）；中风险地区：上海浦东新区，上海青浦区（沪），辽宁省大连（辽B），云南瑞丽市（云N）  2021.12.01消息

新上传的4.py是3.py的升级版，添加了报警模块，识别到来自疫情区域的车牌，会发出蜂鸣器警告，还添加语音播放模块，播放"该车牌来自疫情高风险地区，请注意疫情防控"。
