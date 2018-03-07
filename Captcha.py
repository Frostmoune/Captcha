#coding:utf-8
from urllib import request
from http import cookiejar
from urllib import parse
from PIL import Image
from Learning import *
import time
import urllib
import re

now_path = str(os.getcwd()).replace('\\','/')+"/"

def spider(select):
    if select=="2":
        save_path = now_path + "captcha_test/"
    else:
        save_path = now_path + "captcha_buffer/"
    file_name = now_path + "test.txt"
    # save_path = "G:/Spider/Spider_login/captcha/"
    base_url = "https://uems.sysu.edu.cn/jwxt/#!/login"
    login_url = r"https://cas.sysu.edu.cn/cas/login?service=http%3A%2F%2Fuems.sysu.edu.cn%2Fjwxt%2Fapi%2Fsso%2Fcas%2Flogin%3Fpattern%3Dstudent-login"
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
    if select=="2":
        fp = open(save_path + "result.txt")
        for x in fp.readlines():
            captchas.append(str(x).strip())
        fp.close()
        size = 201
        begin = 101
    else:
        choose = input("\n1、Quick\n2、Accurate\n")
        save_vectors(choose)
        load_all()
        size = 2
        begin = 1
    for i in range(begin,size):
        time.clock()
        now_road = "0"
        if i<100:
            now_road += "0"
        if i<10:
            now_road += "0"
        now_road += str(i) + ".png"
        cookie = cookiejar.MozillaCookieJar(file_name)
        cookie_support = request.HTTPCookieProcessor(cookie)
        opener = request.build_opener(cookie_support)
        response = opener.open(base_url)
        cookie.save(ignore_discard=True, ignore_expires=True)
        req = request.Request(url=captcha_url,headers=headers)
        req.add_header('GET',captcha_url)
        response = opener.open(req)
        data = response.read() 
        fp = open(save_path + now_road,"wb")
        fp.write(data)
        fp.close()
        image = Image.open(save_path + now_road)
        image.show()
        preprocess_image(image,i,select)
        if select=="2":
            captcha = input("请输入验证码\n")
            captchas.append(captcha)
        else:
            predict_image(now_road[:-4],choose)
            print("Time:\t" + str(time.clock()))
    if select=="2":
        fp = open(save_path + "result.txt", "w")
        for x in captchas:
            fp.write(str(x)+"\n")
        fp.close()

def re_preprocess():
    for i in range(1,101):
        save_path = now_path + "captcha_test/"
        now_road = "0"
        if i<100:
            now_road += "0"
        if i<10:
            now_road += "0"
        now_road += str(i) + ".png"
        preprocess_image(Image.open(save_path + now_road),i,"2")

if __name__ == '__main__':
    select = input("1、Test one\n2、Get Data\n3、Test All\n")
    if select!="3":
        spider(select)
    else:
        re_preprocess()
        choose = input("\n1、Quick\n2、Accurate\n")
        save_vectors(choose)
        load_all()
        predict(choose)

