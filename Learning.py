from Preprocessing import *
from svmutil import *
import math

now_path = str(os.getcwd()).replace('\\','/')+"/"
vectors_path = now_path + "divide/"
learn_vectors = {}
model_road = now_path + "captcha_svm/svm_model.model"
model_road_b = now_path + "captcha_svm/svm_model_b.model"
svm_road = now_path + "captcha_svm/"

def load_all(choose_svm):
    if choose_svm == "1":
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

# 求向量点积
def add_vectors(a,b):
    res = 0
    for i in range(len(a)):
        res += float(a[i])*float(b[i])
    return res

# 求向量的模
def module_vectors(a):
    return math.sqrt(sum([float(x)**2 for x in a]))

# 求向量夹角余弦值
def get_cos(a,b):
    add_a, add_b = module_vectors(a),module_vectors(b)
    if add_a!=0 and add_b!=0:
        return add_vectors(a,b)/(add_a*add_b)
    return 0

# 模型的训练函数
def train_model(choose):
    y,x = svm_read_problem(svm_road + "result.txt") # 读取训练数据
    if choose=="2":
        model = svm_train(y,x,'-c 32 -g 0.0078125 -b 1') # -c和-g是与核函数相关的参数，-b 1表示预测结果带概率
        svm_save_model(svm_road + "svm_model.model", model) # 保存模型(特征值为像素值)
    else:
        model = svm_train(y,x,'-c 8.0 -g 8.0 -b 1')
        svm_save_model(svm_road + "svm_model_b.model", model) # 保存模型(特征值为黑点比例)
    return model

# 测试函数（测试多张验证码）
def predict(choose,choose_svm):
    fp = open(now_path + "captcha_test/result.txt")
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
        if choose_svm=="1":
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
        else:
            y_label = [] # 验证码识别的正确答案（若是用于测试，需要读取手动输入的答案，若是仅用于预测，则可初始化为[0,0,0,0]）
            for x in answer:
                if str.isdigit(x):
                    y_label.append(int(x))
                else:
                    y_label.append(ord(x)-87)
            x_value = []
            for vector in all_vectors:
                now_x = {}
                for i in range(1,len(vector)+1):
                    now_x[i] = float(vector[i-1])
                x_value.append(now_x)
            if choose=="2":
                model = svm_load_model(model_road) # 读取模型
            else:
                model = svm_load_model(model_road_b)
            p_label, p_acc, p_val = svm_predict(y_label,x_value,model,'-b 1') #p_label即为预测值
            for x in p_label:
                if int(x)<10:
                    res_str += str(int(x))
                else:
                    res_str += chr(int(x)+87)
        print("Answer:%s\tPredict:%s\t"%(answer,res_str),end="")
        if res_str==answer:
            true_num += 1
            print("True")
        else:
            print("False")
    print("The true rate is:\t"+str(true_num/num))
    fp.close()

# 识别一张验证码
def predict_image(string,select,choose_svm):
    if select=="2":
        all_vectors = divide_image(string,1)
    else:
        all_vectors = divide_image_b(string,1)
    res_str = ""
    if choose_svm=="1":
        for k in range(len(all_vectors)):
            res = 0
            res_key = "Null"
            for key in learn_vectors:
                for i in range(len(learn_vectors[key])):
                    now = get_cos(all_vectors[k],learn_vectors[key][i]) # 计算cos值
                    if now>res: 
                        res = now # 找到最大值
                        res_key = str(key)
            print(res_key + ": " + str(res))
            res_str += str(res_key) # res_str为识别结果
    else:
        y_label = [1,14,1,4]
        x_value = []
        for vector in all_vectors:
            now_x = {}
            for i in range(1,len(vector)+1):
                now_x[i] = int(vector[i-1])
            x_value.append(now_x)
        if select=="2":
            model = svm_load_model(model_road)
        else:
            model = svm_load_model(model_road_b)
        p_label, p_acc, p_val = svm_predict(y_label, x_value, model, '-b 1 -q')
        for x in p_label:
            if int(x)<10:
                res_str += str(int(x))
            else:
                res_str += chr(int(x)+87)
    print("The result is:\t" + res_str)


