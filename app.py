import os
from flask import Flask, render_template_string, request, jsonify
import openai

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# HTML template for the frontend
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot</title>
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
        }
        #user-input {
            width: 100%;
            padding: 10px;
            margin-top: 10px;
        }
        #send-button {
            display: block;
            width: 100%;
            padding: 10px;
            margin-top: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1>Chatbot</h1>
    <div id="chat-container"></div>
    <input type="text" id="user-input" placeholder="Type your message...">
    <button id="send-button">Send</button>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendButton = document.getElementById('send-button');

        function addMessage(message, isUser) {
            const messageElement = document.createElement('p');
            messageElement.textContent = `${isUser ? 'You' : 'Bot'}: ${message}`;
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

        sendButton.addEventListener('click', sendMessage);
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
    
    # Call OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
    )
    
    bot_response = response.choices[0].message['content']
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
