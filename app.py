import os
from flask import Flask, render_template_string, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# HTML template for the frontend
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f0f0;
        }
        #chat-container {
            border: 1px solid #ccc;
            padding: 20px;
            height: 400px;
            overflow-y: scroll;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        #user-input {
            width: calc(100% - 22px);
            padding: 10px;
            margin-top: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        #send-button, #clear-button {
            display: inline-block;
            width: calc(50% - 5px);
            padding: 10px;
            margin-top: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        #clear-button {
            background-color: #f44336;
            margin-left: 10px;
        }
        #send-button:hover, #clear-button:hover {
            opacity: 0.8;
        }
        .message {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e1f5fe;
            text-align: right;
        }
        .bot-message {
            background-color: #f0f0f0;
        }
    </style>
</head>
<body>
    <h1>Interactive Chatbot</h1>
    <div id="chat-container"></div>
    <input type="text" id="user-input" placeholder="Type your message...">
    <button id="send-button">Send</button>
    <button id="clear-button">Clear Conversation</button>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');
        const clearButton = document.getElementById('clear-button');

        function addMessage(message, isUser) {
            const messageElement = document.createElement('div');
            messageElement.className = isUser ? 'message user-message' : 'message bot-message';
            messageElement.textContent = message;
            chatContainer.appendChild(messageElement);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (message) {
                addMessage(message, true);
                userInput.value = '';

                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message }),
                });

                const data = await response.json();
                addMessage(data.response, false);
            }
        }

        function clearConversation() {
            chatContainer.innerHTML = '';
            fetch('/clear', { method: 'POST' });
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

# Store conversation history
conversation_history = []

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": user_message})
    
    # Prepare messages for API call
    messages = [{"role": "system", "content": "You are a helpful assistant."}] + conversation_history
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    bot_response = response.choices[0].message.content.strip()
    
    # Add bot response to conversation history
    conversation_history.append({"role": "assistant", "content": bot_response})
    
    return jsonify({'response': bot_response})

@app.route('/clear', methods=['POST'])
def clear_conversation():
    global conversation_history
    conversation_history = []
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
