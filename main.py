import os
import time
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")

# Configure Pinecone and Google Generative AI
pc = Pinecone(api_key=PINECONE_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
CORS(app)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def run_llm(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    from langchain_community.vectorstores import Pinecone 
    
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings, index_name="aidot")
    print("SEARCH SUCCESSFUL")
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1)
    
    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=docsearch.as_retriever())

    retries = 5
    delay = 2  # Initial delay in seconds

    for i in range(retries):
        try:
            response = qa.invoke(query)
            if response and 'result' in response:
                return response['result']
            else:
                return "No valid response found."
        except Exception as e:
            if 'ResourceExhausted' in str(e):
                print(f"Retrying in {delay} seconds due to resource exhaustion...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise e
    return "Max retries exceeded. Please try again later."

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('message')
    response = run_llm(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
