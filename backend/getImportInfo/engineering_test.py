#!user/bin/env python3
# -*- coding: gbk -*-
import jieba
import os
from docx import Document
from nltk import FreqDist
from nltk import ngrams
# from nltk.book import text6
import nltk
# nltk.download('gutenberg')
#解析抽取函数

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def part_listtopartdict(part_a, part_b): #甲乙方内容一一对应
    engineering_a = part_a
    engineering_b = part_b
    engineering_dict = dict.fromkeys(engineering_a)

    for data_a, data_b in zip(engineering_a, engineering_b):

        if engineering_dict[data_a] == None:
            engineering_dict[data_a] = []
            engineering_dict[data_a].append(data_b)
        else:
            engineering_dict[data_a].append(data_b)

    return engineering_dict


#对单个甲乙方内容的解析
def content_analy1(sentence,contrat_list):
    print(sentence)

    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('./backend/getImportInfo/Userdict.txt')  # 这里加载停用词的路径
    outstr = []
    outstr_new = []
    for word in sentence_seged: #第一次清洗

        if word not in stopwords:

            outstr.append(word)
    # print(outstr)
    contrat_list.append('：')
    contrat_list.append(':')
    outstr_del = []
    for word in outstr: #清除法

        if word in contrat_list:

            outstr_del.append(word)

    for word in outstr_del:
        outstr.remove(word)

    if len(outstr) == 0:
        outstr_new.append(contrat_list[0])
        # outstr_new.append('______')
        # print(outstr_new)
    else:
        outstr_new.append(contrat_list[0])
        # print(outstr)
        outstr_new.extend(outstr)
    # print(contrat_list)
    # print(outstr_new)
    print(outstr_new)
    return outstr_new



def docx_analysis(document):

    # document = Document(docx_path) #获取文档
    Useful_content_list = [] #需要的内容
    contract_jf = [] #甲方
    contract_yf = [] #乙方
    contract_jyf = [] #乙方
    engineering_a = stopwordslist('./backend/getImportInfo/engineering_A.txt')
    engineering_b = stopwordslist('./backend/getImportInfo/engineering_A.txt')
    part_dict = part_listtopartdict(engineering_a,engineering_b)
    # print(part_dict)
    for docx_text in document.paragraphs:#遍历每一个段落

        docu_text = docx_text.text #获取内容
        content_list = docu_text.split('\n') #以换行符分割
        docu_index = int(len(content_list) / 3)
        # print(docu_index)
        if docu_index < 5:
            continue


        for content_index in range(docu_index): #前 6 行 进行关键内容查看
            content = content_list[content_index]
            # print(content)
            content = content.replace(" ", "") #去空格
            for part_a in part_dict.keys():
                if part_a in content and len(contract_jf) == 0: #甲方等关键词信息再且甲方类别还为空
                    for part_b in part_dict[part_a]:
                        if part_b in content and len(contract_yf) == 0: #如果乙方信息也在该行

                            # contract_jyf = content_analy2(content,engineering_a,engineering_b) #解析
                            contract_jf = ('甲方1','______')
                            contract_yf = ('乙方1','______')
                            # print(contract_jf)
                            # print(contract_yf)
                            contract_jyf.extend(contract_jf)
                            contract_jyf.extend(contract_yf)
                            return contract_jyf

                    if  len(contract_jf) == 0: #不在则进行单独抽取
                            # print(contrat_a,contrat_b)

                        contract_jf  = content_analy1(content,engineering_a)


                    for content_index1 in range((content_index+1),docu_index):
                            # print(3)
                        content = content_list[content_index1]
                        print(content)
                        content = content.replace(" ", "")
                        for part_b in part_dict[part_a]:
                            if part_b in content and len(contract_yf) == 0:

                                contract_yf = content_analy1(content, engineering_b)

                                jf = tuple(contract_jf)
                                yf = tuple(contract_yf)

                                Useful_content_list.append(jf)
                                Useful_content_list.append(yf)

                                return Useful_content_list


        for content_index in range(1,content_index): #后6行
            content = content_list[-(content_index)]
            content = content.replace(" ", "")
            if len(contract_jf) != 0:
                for part_b in part_dict[part_a]:
                    if part_b in content and len(contract_yf) == 0:
                        contract_yf = content_analy1(content, engineering_b)

                        jf = tuple(contract_jf)
                        yf = tuple(contract_yf)

                        Useful_content_list.append(jf)
                        Useful_content_list.append(yf)

                        return Useful_content_list
            else :
                for part_a in part_dict.keys():
                    if part_a in content and len(contract_jf) == 0:  # 甲方等关键词信息再且甲方类别还为空
                        for part_b in part_dict[part_a]:
                            if part_b in content and len(contract_yf) == 0:  # 如果乙方信息也在该行

                                # contract_jyf = content_analy2(content,engineering_a,engineering_b) #解析
                                contract_jf = ('甲方1', '______')
                                contract_yf = ('乙方1', '______')
                                # print(contract_jf)
                                # print(contract_yf)
                                contract_jyf.extend(contract_jf)
                                contract_jyf.extend(contract_yf)
                                return contract_jyf

                        if len(contract_jf) == 0:  # 不在则进行单独抽取
                            # print(contrat_a,contrat_b)

                            contract_jf = content_analy1(content, engineering_a)

                        for content_index1 in range((content_index + 1), docu_index):
                            # print(3)
                            content = content_list[content_index1]
                            print(content)
                            content = content.replace(" ", "")
                            for part_b in part_dict[part_a]:
                                if part_b in content and len(contract_yf) == 0:
                                    contract_yf = content_analy1(content, engineering_b)

                                    jf = tuple(contract_jf)
                                    yf = tuple(contract_yf)

                                    Useful_content_list.append(jf)
                                    Useful_content_list.append(yf)

                                    return Useful_content_list



# path='./'
# print(docx_analysis(path))

# 第一步，先获取目录下的文件名列表
# dir_path = './data/'
# jieba.load_userdict("./jieba_dict.txt")
# filepath_list = os.listdir(dir_path)
# content_list = []
# i = 0

# print(filepath_list)
# 第二步，解析每个文档中的内容并进行读取
# for file_path in filepath_list:
#     try:
#         if '~$' in file_path:
#             file_path = file_path.replace('~$','20')
#         docx_path = dir_path + file_path
#         examin_list1 = docx_analysis(docx_path)
#         # print(examin_list1)
#         content_list.extend(examin_list1)
#         # examin_list = ('为了', '提高', '建筑安装', '工程', '的', '建设', '速度', '，', '提高', '经济效益', '，', '经', '________', '建设', '主管部门', '批准', '，', '______', '建设单位', '对', '______', '建筑安装', '工程', '的', '全部', '工程', '或', '单位', '工程', '，', '专业', '工程', '进行', '招标')
#         # if examin_list in examin_list1:
#         #     print(file_path)
#     except :
#         i+=1
#         print('%s抽取失败' % (file_path))
#         pass
#     else :
#         print('%s抽取成功'%(file_path))
#         pass

