import os
from flask import Flask, render_template_string, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Access your API key
api_key = os.getenv('OPENAI_API_KEY')

# Set environment variables
os.environ["OPENAI_API_KEY"] = api_key

# Initialize the ChatOpenAI model
model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.4,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("ai", "{history}")
])

# Initialize conversation memory
memory = ConversationBufferMemory(return_messages=True)

# Initialize the ConversationChain
chain = ConversationChain(
    llm=model,
    memory=memory,
    prompt=prompt
)

# HTML template for the frontend
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KLM-Buddy</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        #chat-container {
            border: 1px solid #ccc;
            padding: 20px;
            height: 400px;
            overflow-y: scroll;
            margin-bottom: 20px;
        }
        #user-input {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
        }
        #send-button, #clear-button {
            padding: 10px;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <h1>KLM-Buddy</h1>
    <p>Ask me bro!!</p>
    <div id="chat-container"></div>
    <input type="text" id="user-input" placeholder="Enter your question...">
    <button id="send-button">Send</button>
    <button id="clear-button">Clear Conversation</button>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');
        const clearButton = document.getElementById('clear-button');

        function addMessage(role, content) {
            const messageElement = document.createElement('p');
            messageElement.innerHTML = `<strong>${role}:</strong> ${content}`;
            chatContainer.appendChild(messageElement);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (message) {
                addMessage('Human', message);
                userInput.value = '';

                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message }),
                });

                const data = await response.json();
                addMessage('AI', data.response);
            }
        }

        async function clearConversation() {
            chatContainer.innerHTML = '';
            await fetch('/clear', { method: 'POST' });
        }

        sendButton.addEventListener('click', sendMessage);
        clearButton.addEventListener('click', clearConversation);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    try:
        # Get the response from the chain
        response = chain.predict(input=user_message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f"An error occurred: {str(e)}"}), 500

@app.route('/clear', methods=['POST'])
def clear_conversation():
    memory.clear()
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
