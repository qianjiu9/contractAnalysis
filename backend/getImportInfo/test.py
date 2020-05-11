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
#������ȡ����
jieba.load_userdict("./backend/getImportInfo/jieba_dict.txt")
# jieba.load_userdict("./jieba_dict.txt")


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def part_listtopartdict(part_a, part_b): #���ҷ�����һһ��Ӧ
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


#�Ե������ҷ����ݵĽ���
def content_analy1(sentence,contrat_list):
    # print(sentence)

    sentence_seged = jieba.cut(sentence.strip())
    # stopwords = stopwordslist('./Userdict.txt')
    stopwords = stopwordslist('./backend/getImportInfo/Userdict.txt')  # �������ͣ�ôʵ�·��
    outstr = []
    outstr_new = []
    for word in sentence_seged: #��һ����ϴ

        if word not in stopwords:

            outstr.append(word)
    # print(outstr)
    contrat_list.append('��')
    contrat_list.append(':')
    outstr_del = []
    for word in outstr: #�����

        if word in contrat_list:

            outstr_del.append(word)

    for word in outstr_del:
        outstr.remove(word)

    if len(outstr) == 0:
        outstr_new.append(contrat_list[0])
        outstr_new.append('______')
        # print(outstr_new)
    else:
        outstr_new.append(contrat_list[0])
        # print(outstr)
        outstr_new.extend(outstr)
    # print(contrat_list)
    # print(outstr_new)
    # print(outstr_new)
    return outstr_new


#�Լ��ҷ�������һ��Ľ���
def content_analy2(sentence,contrat_list1,contrat_list2):

    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('./Userdict.txt')
    stopwords = stopwordslist('./backend/getImportInfo/Userdict.txt')  # �������ͣ�ôʵ�·��
    outstr = []
    for word in sentence_seged:
        # print(word)
        if word not in stopwords:
            outstr.append(word)

    return outstr

def docx_analysis(document):

    # document = Document(docx_path) #��ȡ�ĵ�
    Useful_content_list = [] #��Ҫ������
    contract_jf = [] #�׷�
    contract_yf = [] #�ҷ�
    contract_jyf = [] #�ҷ�
    # engineering_a = stopwordslist('./engineering_A.txt')
    # engineering_b = stopwordslist('./engineering_B.txt')
    engineering_a = stopwordslist('./backend/getImportInfo/engineering_A.txt')
    engineering_b = stopwordslist('./backend/getImportInfo/engineering_B.txt')
    part_dict = part_listtopartdict(engineering_a, engineering_b)
    # print(part_dict)
    for docx_text in document.paragraphs:#����ÿһ������

        docu_text = docx_text.text #��ȡ����
        content_list = docu_text.split('\n') #�Ի��з��ָ�
        docu_index = int(len(content_list) / 3)
        # print(docu_index)
        if docu_index < 5:
            continue


        for content_index in range(docu_index): #ǰ 6 �� ���йؼ����ݲ鿴
            content = content_list[content_index]
            # print(content)
            content = content.replace(" ", "") #ȥ�ո�
            for part_a in part_dict.keys():
                if part_a in content and len(contract_jf) == 0: #�׷��ȹؼ�����Ϣ���Ҽ׷����Ϊ��
                    for part_b in part_dict[part_a]:
                        if part_b in content and len(contract_yf) == 0: #����ҷ���ϢҲ�ڸ���

                            # contract_jyf = content_analy2(content,engineering_a,engineering_b) #����
                            contract_jf = ('�׷�1','______')
                            contract_yf = ('�ҷ�1','______')
                            contract_jyf.extend(contract_jf)
                            contract_jyf.extend(contract_yf)
                            return contract_jyf

                    if  len(contract_jf) == 0: #��������е�����ȡ
                            # print(contrat_a,contrat_b)

                        contract_jf  = content_analy1(content,engineering_a)


                    for content_index1 in range((content_index+1),docu_index):
                            # print(3)
                        content = content_list[content_index1]
                        # print(content)
                        content = content.replace(" ", "")
                        for part_b in part_dict[part_a]:
                            if part_b in content and len(contract_yf) == 0:

                                contract_yf = content_analy1(content, engineering_b)

                                jf = tuple(contract_jf)
                                yf = tuple(contract_yf)

                                Useful_content_list.append(jf)
                                Useful_content_list.append(yf)
                                # print(Useful_content_list)

                                return Useful_content_list


        for content_index in range(1,content_index): #��6��
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
                    if part_a in content and len(contract_jf) == 0:  # �׷��ȹؼ�����Ϣ���Ҽ׷����Ϊ��
                        for part_b in part_dict[part_a]:
                            if part_b in content and len(contract_yf) == 0:  # ����ҷ���ϢҲ�ڸ���

                                contract_jyf = content_analy2(content,engineering_a,engineering_b) #����
                                contract_jf = ('�׷�1', '______')
                                contract_yf = ('�ҷ�1', '______')
                                # print(contract_jf)
                                # print(contract_yf)
                                contract_jyf.extend(contract_jf)
                                contract_jyf.extend(contract_yf)
                                return contract_jyf

                        if len(contract_jf) == 0:  # ��������е�����ȡ
                            # print(contrat_a,contrat_b)

                            contract_jf = content_analy1(content, engineering_a)

                        for content_index1 in range((content_index + 1), docu_index):
                            # print(3)
                            content = content_list[content_index1]
                            # print(content)
                            content = content.replace(" ", "")
                            for part_b in part_dict[part_a]:
                                if part_b in content and len(contract_yf) == 0:
                                    contract_yf = content_analy1(content, engineering_b)

                                    jf = tuple(contract_jf)
                                    yf = tuple(contract_yf)

                                    Useful_content_list.append(jf)
                                    Useful_content_list.append(yf)

                                    return Useful_content_list






# ��һ�����Ȼ�ȡĿ¼�µ��ļ����б�
# dir_path = './data/'
# jieba.load_userdict("./jieba_dict.txt")
# filepath_list = os.listdir(dir_path)
# content_list = []
# i = 0
# print(filepath_list)
# �ڶ���������ÿ���ĵ��е����ݲ����ж�ȡ
# for file_path in filepath_list:
#     try:
#         if '~$' in file_path:
#             file_path = file_path.replace('~$','20')
#         docx_path = dir_path + file_path
#         examin_list1 = docx_analysis(docx_path)
#         # print(examin_list1)
#         content_list.extend(examin_list1)
#         examin_list = ('�׷�1',)
#         if examin_list in examin_list1:
#             print(file_path)
#     except :
#         i+=1
#         print('%s��ȡʧ��' % (file_path))
#         pass
#     else :
#         # print('%s��ȡ�ɹ�'%(file_path))
#         pass
# print('��ȡʧ��%d��' % (i))
# #��������ͨ���ڶ����õ��ķִ��б����жϳ�ȡ����
# bigrams = ngrams(content_list, 1)
# # print(bigrams)
# bigramsDist = FreqDist(bigrams)
# print(bigramsDist.most_common(1000))
# for data in bigramsDist.most_common(20):
#     print(data)
#     print(data[0][0])
#     print(data[1])

# print(docx_analysis('./2015����װ�޺�ͬ������.docx'))
