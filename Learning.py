from Preprocessing import *
import math

now_path = str(os.getcwd()).replace('\\','/')+"/"
vectors_path = now_path + "divide/"
learn_vectors = {}
def load_all():
    for i in range(26):
        fp = open(vectors_path + chr(i+97) + "/vectors.txt")
        learn_vectors[chr(i+97)] = []
        for x in fp.readlines():
            learn_vectors[chr(i+97)].append(str(x).strip().split(" "))
        fp.close()
    for i in range(10):
        fp = open(vectors_path + str(i) + "/vectors.txt")
        learn_vectors[str(i)] = []
        for x in fp.readlines():
            learn_vectors[str(i)].append(str(x).strip().split(" "))
        fp.close()

def add_vectors(a,b):
    res = 0
    for i in range(len(a)):
        res += float(a[i])*float(b[i])
    return res

def module_vectors(a):
    return math.sqrt(sum([float(x)**2 for x in a]))

def get_cos(a,b):
    add_a, add_b = module_vectors(a),module_vectors(b)
    if add_a!=0 and add_b!=0:
        return add_vectors(a,b)/(add_a*add_b)
    return 0

def preprocess_image(image,num,select):
    now_road = "0"
    if num<100:
        now_road += "0"
    if num<10:
        now_road += "0"
    now_road += str(num)
    pre_processing_image(image,now_road,select)

def predict(choose):
    fp = open(now_path + "result.txt")
    num = 0
    true_num = 0
    for x in fp.readlines():
        res_str = ""
        num += 1
        answer = str(x).strip()
        now_road = "0"
        if num<100:
            now_road += "0"
        if num<10:
            now_road += "0"
        now_road += str(num)
        if choose=="2":
            all_vectors = divide_image(now_road,2)
        else:
            all_vectors = divide_image_b(now_road,2)
        for k in range(len(all_vectors)):
            res = 0
            res_key = "Null"
            for key in learn_vectors:
                for i in range(len(learn_vectors[key])):
                    now = get_cos(all_vectors[k],learn_vectors[key][i])
                    if now>res:
                        res = now
                        res_key = str(key)
            res_str += str(res_key)
        print("Answer:%s\tPredict:%s\t"%(answer,res_str),end="")
        if res_str==answer:
            true_num += 1
            print("True")
        else:
            print("False")
    print("The true rate is:\t"+str(true_num/num))
    fp.close()

def predict_image(string,select):
    if select=="2":
        all_vectors = divide_image(string,1)
    else:
        all_vectors = divide_image_b(string,1)
    res_str = ""
    for k in range(len(all_vectors)):
        res = 0
        res_key = "Null"
        for key in learn_vectors:
            for i in range(len(learn_vectors[key])):
                now = get_cos(all_vectors[k],learn_vectors[key][i])
                if now>res:
                    res = now
                    res_key = str(key)
        print(res_key + ": " + str(res))
        res_str += str(res_key)
    print(res_str)

