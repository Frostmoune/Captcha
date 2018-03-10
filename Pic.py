from PIL import Image
from PIL import ImageDraw
import os

now_path = str(os.getcwd()).replace('\\','/') + "/" #得到当前目录

# 判断是否为同一种类的颜色
def isSameColor(a,b):
    return abs(a[0]-b[0])<10 and abs(a[1]-b[1])<10 and abs(a[2]-b[2])<10 

# 得到不同种类颜色的坐标
def divideColor(image):
    color_map = {}
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            now_color = image.getpixel((i,j)) # 得到该点的RGB值，注意参数是一个元组
            flag = 1
            for key in color_map:
                if isSameColor(key,now_color):
                    flag = 0
                    color_map[key].append((i,j)) # 若是相同种类的颜色，记录当前坐标
            if flag:
                color_map[now_color] = [(i,j)] # 否则添加一种新的颜色
    return color_map

# 在空白图上画点
def divideDraw(color_map):
    save_path = now_path + "pixel/" # 新图的存储目录
    for key in color_map:
        now_image = Image.new('RGB',(90,32),(255,255,255)) # 新建一张图，它是RGB类型（第一个参数为类型）的、
                                                           # 尺寸为90*32（第二个参数为尺寸）、背景为白色（第三个参数为背景色）
        drawer = ImageDraw.Draw(now_image)# 新建画笔
        for x in color_map[key]:
            drawer.point(x,key)# 描点画图
        now_image.save(save_path + str(key) + ".png")# 将得到的图存起来

test_image = Image.open(now_path + "captcha/0001.png")# 打开一张验证码
divideDraw(divideColor(test_image))
