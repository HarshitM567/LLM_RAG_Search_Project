
from flask import Flask, request, jsonify
from utils import search_articles, concatenate_content, generate_answer

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get("query")
    print("Received query:", user_query)

    articles = search_articles(user_query)
    content = concatenate_content(articles)
    answer = generate_answer(content, user_query)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='localhost', port=5001)
