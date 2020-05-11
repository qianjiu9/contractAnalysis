from flask import render_template, Blueprint, escape, request, jsonify
import json
from docx import Document
from backend.prediction.predict import predict_type
from backend.getImportInfo.test import docx_analysis
from pymongo import MongoClient
from bson import json_util, ObjectId
from data_helpers import clean_chars
import re
import time
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher

graph = Graph("http://localhost:7474", auth=("neo4j", "zzl1997"))

main = Blueprint('main', __name__, template_folder='templates', static_folder='static', static_url_path="/static")
conn = MongoClient('localhost', 27017)
db = conn.constract


@main.route('/', defaults={'path': ''})
@main.route('/<path:path>')
def index(path):
    return render_template('index.html')


types = {'sale': '销售合同',
         'borrow': '借款合同',
         'labor': '劳务合同',
         'engineer': '项目合同',
         'business': '合作合同',
         'other': '其他'}
rela = {
    '劳务合同': ['聘用', '就职'],
    '销售合同': ['卖给', '买入'],
    '借款合同': ['借出', '借入'],
    '项目合同': ['委托', '代理'],
    '合作合同': ['合作', '合作']
}


@main.route('/querybyid/<id>', methods=['get'])
def query_by_id(id):
    if request.method == 'GET':
        document = list(db.documents.find({"_id": ObjectId(id)}))
        relations = graph.run("MATCH p = (c1: company {name: '一企签'}) - [:合作]->(c) RETURN relationships(p), c,c1").data()
        print(list(relations))
        res = {'document': document, 'relations': list(relations)}
        return json.dumps(res, default=json_util.default)


@main.route('/query-node-graph', methods=['post'])
def query_node_graph():
    if request.method == 'POST':
        data = json.loads(request.get_data(as_text=True))
        name = data['name']
        query1 = "MATCH p = (p1:partA" + "{name:'" + name + "'}) - [r] ->(p2) RETURN relationships(p),p1,p2"
        query2 = "MATCH p = (p1:partB" + "{name:'" + name + "'}) - [r] ->(p2) RETURN relationships(p),p1,p2"
        relation1 = graph.run(query1).data()
        relation2 = graph.run(query2).data()
        res = relation1 + relation2
        print(query1)
        print(query2)
        print(res)

        return json.dumps(res)



@main.route('/query-graph', methods=['post'])
def query_graph():
    if request.method == 'POST':
        get_data = json.loads(request.get_data(as_text=True))
        part_a = get_data['partA']
        part_b = get_data['partB']
        type = get_data['type']
        re1 = rela[type][0]
        re2 = rela[type][1]
        query1 = "MATCH p = (p1: partA {name:'" + part_a + "'}) - [:" + re1 + "] ->(p2) RETURN relationships(p),p1,p2"
        query2 = "MATCH p = (p1: partB {name:'" + part_b + "'}) - [:" + re2 + "] ->(p2) return relationships(p),p1,p2"
        # query2 = "MATCH p = (p1) - [:" + re1 + "] ->(p2: partB {name:'" + part_b + "'} ) return relationships(p),p1,p2"
        print(query1)
        print(query2)
        relation1 = graph.run(query1).data()
        relation2 = graph.run(query2).data()
        res = relation1 + relation2
        return json.dumps(res)


@main.route('/updatebyid/<id>', methods=['get', 'post'])
def update_by_id(id):
    if request.method == 'POST':
        get_data = json.loads(request.get_data(as_text=True))
        db.documents.update_one({'_id': ObjectId(id)},
                                {"$set": {'partA': get_data.get('partA'), 'partB': get_data.get('partB')}})

        n1 = Node("partA", name=get_data.get('partA'))
        n2 = Node("partB", name=get_data.get('partB'))
        relas = rela[get_data['type']]
        tx = graph.begin()
        rel_props = {
            'name': relas[0]
        }
        rel_props1 = {
            'name': relas[1]
        }
        rel = Relationship(n1, relas[0], n2, **rel_props)
        rel2 = Relationship(n2, relas[1], n1, **rel_props1)
        tx.merge(n1, 'partA', 'name')
        tx.merge(n2, 'partB', 'name')
        tx.merge(rel)
        tx.merge(rel2)
        tx.commit()
        return jsonify({'status': 200, 'msg': 'success'})


@main.route('/file-list', methods=['get'])
def file_list():
    type = request.args.get('type')
    name = request.args.get('name')
    if type is None and name is None:
        params = {}
    elif type is not None and name is not None:
        params = {'type': types[type], 'name': re.compile(name)}
    elif name is None:
        params = {'type': types[type]}
    else:
        params = {'name': re.compile(name)}
    docs_list = list(db.documents.find(params).sort([('date', -1)]))

    return json.dumps(docs_list, default=json_util.default)


@main.route('/upload', methods=['post'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        name = f.filename
        text = Document(f)
        res = docx_analysis(text)
        partA = clean_chars(''.join(res[0][1:len(res[0])]))
        partB = clean_chars(''.join(res[1][1:len(res[1])]))
        print(res)
        print(partA)
        print(partB)
        parlist = []
        for par in text.paragraphs:
            parlist.append(par.text)
        document = '\n'.join(parlist)
        # print(document)
        type = predict_type(document)
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        info = []

        db.documents.insert({'name': name, 'text': document, 'type': type,
                             'info': info, 'date': date, 'partA': partA, 'partB': partB, 'signDate': ''})
        print(type)
        if type != '其他合同':
            merge1 = "MERGE (p1: partA{name:'" + partA + "'}) - [:" + rela[type][0] + "{name: '" + rela[type][0] + "'}] ->(p2: partB{name:'"+ partB+"'})"
            merge2 = "MERGE (p1: partB{name:'" + partB + "'}) - [:" + rela[type][1] + "{name: '" + rela[type][1] + "'}] ->(p2: partA{name:'" + partA + "'})"
            print(merge1)
            print(merge2)
            graph.run(merge1)
            graph.run(merge2)
        return jsonify({'code': 200, 'msg': '请求成功'})


@main.route('/delete-document/<id>', methods=['delete'])
def delete_document(id=None):
    if request.method == 'DELETE':
        db.documents.remove({"_id": ObjectId(id)})
        return jsonify({'code': 200, 'msg': '删除成功'})
