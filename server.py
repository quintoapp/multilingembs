from flask import Flask, request
from flask_cors import CORS, cross_origin
import ujson as json
import utils

app = Flask(__name__)
CORS(app)

@app.route('/getEmbeddings',methods=['POST'])
def getEmbeddings():
	data = json.loads(request.data)
	ret_obj = []
	# src_lang = data['src_lang']
	# src_sent = data['src_sent']
	# trgt_lang = data['trgt_lang']
	# trgt_sent = data['trgt_sent']
	# src_vec = utils.fetch_vectors(src_lang, src_sent)
	# trgt_vec = utils.fetch_vectors(trgt_lang, trgt_sent)
	for idx, d in enumerate(data):
		sent_key = 'S' + str(idx+1)
		l_key = 'L' + str(idx+1)
		sent = d[sent_key]
		lang = d[l_key]
		ret_obj.append({sent_key + '_embeddings':utils.fetch_vectors(lang, sent)})
	#print(src_vec)
	# sim = utils.similarity(src_vec, trgt_vec)
	# return json.dumps({"src_vec": src_vec, "trgt_vec": trgt_vec, "similarity":sim})
	#return json.dumps({"msg":"ok"})
	return json.dumps(ret_obj)

if __name__ == '__main__':
	app.run(host="0.0.0.0",port=8080, debug = True)
