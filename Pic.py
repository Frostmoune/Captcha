from PIL import Image
from PIL import ImageDraw
import os

now_path = str(os.getcwd()).replace('\\','/') + "/"

def isSameColor(a,b):
    return abs(a[0]-b[0])<10 and abs(a[1]-b[1])<10 and abs(a[2]-b[2])<10

def divideColor(image):
    color_map = {}
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            now_color = image.getpixel((i,j))
            flag = 1
            for key in color_map:
                if isSameColor(key,now_color):
                    flag = 0
                    color_map[key].append((i,j))
            if flag:
                color_map[now_color] = [(i,j)]
    return color_map

def divideDraw(color_map):
    save_path = now_path + "pixel/"
    for key in color_map:
        now_image = Image.new('RGB',(90,32),(255,255,255))
        drawer = ImageDraw.Draw(now_image)
        for x in color_map[key]:
            drawer.point(x,key)
        now_image.save(save_path + str(key) + ".png")

test_image = Image.open(now_path + "captcha/0001.png")
divideDraw(divideColor(test_image))
