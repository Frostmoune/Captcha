import os
from PIL import Image
from PIL import ImageDraw

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
    return table

def get_pixel(image):
    fp = open("test.txt","w")
    for i in range(0,image.size[0]):
        for j in range(0,image.size[1]):
            fp.write(str(image.getpixel((i,j)))+" ")
        fp.write("\n")
    fp.close()

def isvalid(image, x, y):
    if x<0 or x>=image.size[0] or y<0 or y>=image.size[1]:
        return False
    return True

def clear_noise_pixel_binary(image, x, y, after_table,level):
    now = image.getpixel((x,y))
    flag = 0
    for i in range(-1,2):
        for j in range(-1,2):
            if i==0 and j==0:
                continue
            if isvalid(image, x+i, y+j):
                if image.getpixel((x+i,y+j))==0:
                    flag+=1
    if now==0 and flag<level:
        after_table_b[y][x] = 1
    elif now==1 and flag>=4:
        after_table_b[y][x] = 0
    else:
        after_table_b[y][x] = now

def isblack(obj):
    return obj[0]<=25 and obj[1]<=25 and obj[2]<=25

def islight(obj):
    return obj[0]>=245 or obj[1]>=245 or obj[2]>=245

def clear_noise_pixel_b(image, x, y, after_table, level):
    now = image.getpixel((x,y))
    flag = 0
    for i in range(-1,2):
        for j in range(-1,2):
            if i==0 and j==0:
                continue
            if isvalid(image, x+i, y+j):
                if isblack(now):
                    flag += 1
    if flag<level:
        after_table_b[y][x] = now
    else:
        after_table_b[y][x] = (255,255,255)

def clear_black(image, x, y, after_table_b):
    now = image.getpixel((x,y))
    if now[0]<=15 and now[1]<=15 and now[2]<=15:
        after_table_b[y][x] = (255,255,255)
    else: 
        after_table_b[y][x] = now

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
read_path = now_path + "captcha/"
need_path = now_path + "captcha_need/"
divide_path = now_path + "divide/"
test_road = now_path + "captcha_test/"
buffer_road = now_path + "captcha_buffer/"
table = get_bin_table()

def pre_processing():
    for i in range(1,len(os.listdir(read_path))):
        now_road = "0"
        if i<100:
            now_road += "0"
        if i<10:
            now_road += "0"
        now_road += str(i) + ".png"
        print(now_road)
        now_image = Image.open(read_path + now_road)
        clear_noise(now_image,2)
        now_image = now_image.convert("L")
        bin_image = now_image.point(table, '1')
        clear_noise(bin_image,1)
        bin_image.resize((64, 24),Image.ANTIALIAS).save(need_path + now_road)

def get_all_pixel(image):
    res = []
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            if image.getpixel((i,j))==255:
                res.append("1")
            else:
                res.append("0")
    return res

def get_all_eigen_b(image):
    res = [0 for i in range(17)]
    sum_pixel = 0
    for i in range(4):
        for j in range(4):
            now_image = image.crop((j*4,i*4,(j+1)*4,(i+1)*4))
            now_pixel = 0
            for x in range(now_image.size[0]):
                for y in range(now_image.size[1]):
                    if now_image.getpixel((x,y))==0:
                        now_pixel += 1
            res[i*4+j] = now_pixel/24
            sum_pixel += now_pixel
    res[16] = sum_pixel/384
    return res

char_vectors = {}

def divide_pic():
    for i in range(26):
        if not os.path.exists(divide_path + chr(i+97)):
            os.mkdir(divide_path + chr(i+97))
    for j in range(10):
        char_vectors[str(j)] = []
        if not os.path.exists(divide_path + str(j)):
            os.mkdir(divide_path + str(j))
    fp = open(read_path + "/result.txt")
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
        now_image = Image.open(need_path + read_road)
        for j in range(4):
            child_image = now_image.crop((j*16,0,(j+1)*16,24))
            write_road = now_road + str(i) + "-" + str(j) + ".png"
            child_image.save(divide_path + divide_name[i-1][j] + "/" + write_road)

def save_vectors(select):
    for i in range(26):
        char_vectors[chr(i+97)] = []
    for j in range(10):
        char_vectors[str(j)] = []
    for key in char_vectors:
        for x in os.listdir(divide_path + key):
            now_png = str(x)
            if now_png[-4:]==".png":
                image = Image.open(divide_path + key + "/" + now_png, "r")
                if select=="2":
                    char_vectors[key].append(get_all_pixel(image))
                else:
                    char_vectors[key].append(get_all_eigen_b(image))
        fp = open(divide_path + key + "/vectors.txt", "w")
        for i in range(len(char_vectors[key])):
            for j in range(len(char_vectors[key][i])):
                fp.write(str(char_vectors[key][i][j]))
                fp.write(" ")
            fp.write("\n")
        fp.close()

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