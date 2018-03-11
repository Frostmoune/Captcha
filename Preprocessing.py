import os
from PIL import Image
from PIL import ImageDraw
from svmutil import *

after_table = [[0 for x in range(64)] for x in range(24)]
after_table_b = [[0 for x in range(90)] for x in range(32)]
steps = [[-1,0],[0,-1],[1,0],[0,1]]

def get_bin_table(thresold = 170):
    table = []
    for i in range(256):
        if i < thresold:
            table.append(0)
        else:
            table.append(1)
    return table # 得到的一个list，其0~thresold-1项为0，thresold~255项为1

# 得到图像中所有点的特征值
def get_pixel(image):
    fp = open("test.txt","w")
    for i in range(0,image.size[0]):
        for j in range(0,image.size[1]):
            fp.write(str(image.getpixel((i,j)))+" ")
        fp.write("\n")
    fp.close()

# 判断某个点是否超出了图的边界
def isvalid(image, x, y):
    if x<0 or x>=image.size[0] or y<0 or y>=image.size[1]:
        return False
    return True

# 判断某个点是否为噪点，after_table_b用于描点画图，可以改变level以调节去噪深度
def clear_noise_pixel_binary(image, x, y, after_table_b, level):
    now = image.getpixel((x,y))
    flag = 0
    for i in range(-1,2):
        for j in range(-1,2):
            if i==0 and j==0:
                continue
            if isvalid(image, x+i, y+j):
                if image.getpixel((x+i,y+j))==0:
                    flag+=1 # 计算该点周围黑色点的数量
    if now==0 and flag<level:
        after_table_b[y][x] = 1 # 去除操作，若该点为黑点，且周围黑点的数量小于level，则将该点变为白点
    elif now==1 and flag>=4:
        after_table_b[y][x] = 0 # 补充操作，若该点为白点，且周围黑点的数量大于等于4，则将该点变为黑点
    else:
        after_table_b[y][x] = now

def isblack(obj):
    return obj[0]<=25 and obj[1]<=25 and obj[2]<=25

def islight(obj):
    return obj[0]>=245 or obj[1]>=245 or obj[2]>=245

# 清除所有黑色点，after_table_b是用于描点画图的
def clear_black(image, x, y, after_table_b):
    now = image.getpixel((x,y)) 
    if now[0]<=15 and now[1]<=15 and now[2]<=15:
        after_table_b[y][x] = (255,255,255) #如果是该像素点是黑色的，则直接将它暴力设置为白色
    else: 
        after_table_b[y][x] = now #否则就设置成它原来的颜色

# 总的去噪函数
def clear_noise(image,select=0):
    draw = ImageDraw.Draw(image)
    for k in range(1):
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                if select==1:
                    clear_noise_pixel_binary(image,i,j,after_table_b,2)
                else:
                    clear_black(image,i,j,after_table_b)
    try:
        for i in range(image.size[0]):
            for j in range(image.size[1]):
                draw.point((i,j),after_table_b[j][i])
    except Exception as e:
        print(e)

now_path = str(os.getcwd()).replace('\\','/')+"/"
read_path = now_path + "captcha/" # 验证码的原始路径
first_path = now_path + "captcha_first/" # 去除了干扰线的验证码的储存路径
gray_path = now_path + "captcha_gray/" # 灰度图的存储路径
binary_path = now_path + "captcha_binary/" # 二值图的存储路径
need_path = now_path + "captcha_need/" # 最后得到的验证码路径
divide_path = now_path + "divide/" # 分割后的验证码路径
test_road = now_path + "captcha_test/" # 测试样例路径
buffer_road = now_path + "captcha_buffer/"
svm_road = now_path + "captcha_svm/" # svm模型路径
table = get_bin_table() # thresold的值可以自行调节

def pre_processing():
    for i in range(1,len(os.listdir(read_path))):
        now_road = "0"
        if i<100:
            now_road += "0"
        if i<10:
            now_road += "0"
        now_road += str(i) + ".png"
        print(now_road)
        now_image = Image.open(read_path + now_road) # 打开一张图片
        clear_noise(now_image,2) # 第一步，去除干扰线
        now_image.save(first_path + now_road) # 可省略，存储清除了干扰线的图片 
        now_image = now_image.convert("L") # "L"表示灰度图
        now_image.save(gray_path + now_road) # 可省略，存储灰度图
        bin_image = now_image.point(table, '1') # 用重新描点画图的方式得到二值图
        bin_image.save(binary_path + now_road) # 可省略，存储二值图
        clear_noise(bin_image,1) # 最后一步，对二值图去噪
        bin_image.resize((64, 24),Image.ANTIALIAS).save(need_path + now_road) # 改变图片的分辨率后，将最终的图片存储下来

def get_all_pixel(image):
    res = []
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            if image.getpixel((i,j))==255:
                res.append("1")
            else:
                res.append("0")
    return res

# 根据选取规则得到图像特征值
def get_all_eigen_b(image):
    res = [0 for i in range(17)]
    sum_pixel = 0
    for i in range(4):
        for j in range(4):
            now_image = image.crop((j*4,i*6,(j+1)*4,(i+1)*6)) # 分割图像
            now_pixel = 0
            for x in range(now_image.size[0]):
                for y in range(now_image.size[1]):
                    if now_image.getpixel((x,y))==0:
                        now_pixel += 1 # 计算黑色点数量
            res[i*4+j] = now_pixel/24 # 计算黑色点比例
            sum_pixel += now_pixel
    res[16] = sum_pixel/384
    return res

char_vectors = {}

def divide_pic():
    # 在路径下新建文件夹，名字为a-z,0-9,用于存储分割后的验证码
    for i in range(26):
        if not os.path.exists(divide_path + chr(i+97)):
            os.mkdir(divide_path + chr(i+97))
    for j in range(10):
        char_vectors[str(j)] = []
        if not os.path.exists(divide_path + str(j)):
            os.mkdir(divide_path + str(j))
    fp = open(read_path + "/result.txt") # 验证码训练集的答案路径
    divide_name = []
    for x in fp.readlines():
        divide_name.append(str(x).strip())
    fp.close()
    for i in range(1,len(os.listdir(need_path))):
        now_road = "/0"
        if i<100:
            now_road += "0"
        if i<10:
            now_road += "0"
        read_road = now_road + str(i) + ".png"
        now_image = Image.open(need_path + read_road) # 读取处理后的验证码
        for j in range(4):# 每张验证码有四个字符
            child_image = now_image.crop((j*16,0,(j+1)*16,24)) # 分割验证码图片（均分）
            write_road = now_road + str(i) + "-" + str(j) + ".png"
            child_image.save(divide_path + divide_name[i-1][j] + "/" + write_road) # 存储分割后的图片

def save_vectors(select):
    for i in range(26):
        char_vectors[chr(i+97)] = []
    for j in range(10):
        char_vectors[str(j)] = [] # 存放不同字符及其对应的特征向量
    for key in char_vectors:
        for x in os.listdir(divide_path + key):
            now_png = str(x)
            if now_png[-4:]==".png":
                image = Image.open(divide_path + key + "/" + now_png, "r") # 打开一张图片
                if select=="2":
                    char_vectors[key].append(get_all_pixel(image)) # 像素值作为特征值
                else:
                    char_vectors[key].append(get_all_eigen_b(image)) # 黑点比例作为特征值
        fp = open(divide_path + key + "/vectors.txt", "w") # 保存
        for i in range(len(char_vectors[key])):
            for j in range(len(char_vectors[key][i])):
                fp.write(str(char_vectors[key][i][j]))
                fp.write(" ")
            fp.write("\n")
        fp.close()
    fp = open(svm_road + "result.txt","w")
    for key in char_vectors:
        now = str(key)
        if str.isdigit(now):
            value = int(now)
        else:
            value = ord(now)-87
        for i in range(len(char_vectors[key])):
            fp.write(str(value))
            fp.write(" ")
            num = 1
            for x in char_vectors[key][i]:
                fp.write("%d:%lf "%(num,float(x)))
                num += 1
            fp.write("\n")
    fp.close()

# 预处理单个图
def pre_processing_image(image,string,select):
    now_image = image
    clear_noise(now_image,2)
    now_image = now_image.convert("L")
    bin_image = now_image.point(table, '1')
    clear_noise(bin_image, 1)
    if select=="2":
        road = test_road + string + "_test.png"
    else:
        road = buffer_road + string + "_test.png"
    bin_image.resize((64, 24),Image.ANTIALIAS).save(road)

# 分割单个图，将其所有点的像素值作为特征点
def divide_image(string,select):
    now_vectors = []
    if select==2:
        image = Image.open(test_road + string + "_test.png")
    else:
        image = Image.open(buffer_road + string + "_test.png")
    for j in range(4):
        child_image = image.crop((j*16,0,(j+1)*16,24))
        now_vectors.append(get_all_pixel(child_image))
    return now_vectors

# 分割单个图，将设定的特征值作为该图特征值
def divide_image_b(string,select):
    now_vectors = []
    if select==2:
        image = Image.open(test_road + string + "_test.png")
    else:
        image = Image.open(buffer_road + string + "_test.png")
    for j in range(4):
        child_image = image.crop((j*16,0,(j+1)*16,24))
        now_vectors.append(get_all_eigen_b(child_image))
    return now_vectors