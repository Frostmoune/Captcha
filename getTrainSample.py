from urllib import request
from http import cookiejar
from urllib import parse
from PIL import Image
from Preprocessing import *

NOW_PATH = str(os.getcwd()).replace('\\','/') + "/"
SAVE_PATH = NOW_PATH + 'captcha/'

def spider():
    base_url = "https://uems.sysu.edu.cn/jwxt/#!/login"
    captcha_url = "https://cas.sysu.edu.cn/cas/captcha.jsp"
    headers = {}
    headers['Accept'] = "image/webp,image/apng,image/*,*/*;q=0.8"
    headers['Accept-Encoding'] = "gzip, deflate"
    headers['Accept-Language'] = "zh-CN,zh;q=0.9"
    headers['Connection'] = "keep-alive"
    headers['Host'] = "cas.sysu.edu.cn"
    headers['Referer'] = r"https://cas.sysu.edu.cn/cas/login?service=http%3A%2F%2Fuems.sysu.edu.cn%2Fjwxt%2Fapi%2Fsso%2Fcas%2Flogin%3Fpattern%3Dstudent-login"
    headers['User-Agent'] = r'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.119 Safari/537.36'
    captchas = []

    fp = open(SAVE_PATH + "result.txt")
    for x in fp.readlines():
        captchas.append(str(x).strip())
    fp.close()

    begin = len(os.listdir(SAVE_PATH))
    file_name = now_path + "test.txt"
    for i in range(begin, begin + 1000):
        now_road = "0"
        if i < 100:
            now_road += "0"
        if i < 10:
            now_road += "0"
        now_road += str(i) + ".png"

        cookie = cookiejar.MozillaCookieJar(file_name)
        cookie_support = request.HTTPCookieProcessor(cookie)
        opener = request.build_opener(cookie_support)
        response = opener.open(base_url)
        cookie.save(ignore_discard = True, ignore_expires = True)
        req = request.Request(url = captcha_url, headers = headers)
        req.add_header('GET', captcha_url)
        response = opener.open(req)
        data = response.read()
        
        fp = open(SAVE_PATH + now_road, "wb")
        fp.write(data)
        fp.close()
        image = Image.open(SAVE_PATH + now_road)
        image.show()
        captcha = input("请输入验证码\n")
        if captcha == '-1':
            os.remove(SAVE_PATH + now_road)
            break
        captchas.append(captcha)

    fp = open(SAVE_PATH + "result.txt", "w")
    for x in captchas:
        fp.write(str(x) + "\n")
    fp.close()
    print("储存完成...")

    print("开始预处理数据集...")
    pre_processing()
    divide_pic()

# spider()
# divide_pic()