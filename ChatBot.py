import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from flask import Flask, redirect, url_for, render_template, request
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Set Groq API key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

# Initialize the chat model with the Groq API key
groq = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="mixtral-8x7b-32768")

# Load and split documents
loader = CSVLoader(file_path='./Company_data.csv',
                   csv_args={
                       "delimiter": ",",
                       "quotechar": '"',
                   },)

data = loader.load()


documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(data)

# Initialize the embeddings model with the Google Generative AI API key
embeddings = GoogleGenerativeAIEmbeddings(api_key=os.getenv('GOOGLE_API_KEY'), model="models/embedding-001")
vectordb = FAISS.from_documents(data, embedding=embeddings)
vectordb.save_local("faiss_index")

prompt = ChatPromptTemplate.from_template("""
Context: HnH Tech Solutions is a company specializing in designing and engineering software using mobile, web, 
and cloud technologies. They focus on custom web applications, mobile apps, and data mining techniques. 
Your task is to create a chatbot that can maintain a conversation with users. The chatbot should provide accurate answers based 
on the provided context, search the context thoroughly before answering, think step-by-step before delivering a detailed answer, 
and not include any additional information such as company data unless specifically asked for.

Instructions:

The chatbot should be able to respond to a variety of questions related to the company's services, technologies, approach, and more.
Ensure that the chatbot's responses are detailed and informative, providing valuable insights to the users.
If the requested information is not available in the provided context, the chatbot should respond with "The information you are asking for is not available in the company data."

<context>
{context}
</context>
                                          
Question: {input}
""")

# Create the document chain
document_chain = create_stuff_documents_chain(groq, prompt)

# Retriever
retriever = vectordb.as_retriever()

# Retrieval Chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Define a memory with a buffer size of 2 to store the last 2 responses
memory = ConversationBufferMemory()


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    # Invoke retrieval chain with input and context
    response = retrieval_chain.invoke({"input":input})
    answer = response.get('answer')
    
    # Store input and answer in memory
    memory.chat_memory.add_user_message(input)
    memory.chat_memory.add_ai_message(answer)

    print("Response: ", answer)
    return str(answer)


if __name__ == '__main__':
    app.run(debug=True)

