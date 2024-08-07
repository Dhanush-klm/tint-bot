import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up ChatOpenAI model
model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.4,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}")
    ]
)

# Combine the prompt template and the model
chain = prompt | model

class Query(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(query: Query):
    try:
        input_data = {"input": query.message}
        response = chain.invoke(input_data)
        return {"response": response.content}
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return """
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
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                margin-bottom: 20px;
            }
            #user-input {
                width: 100%;
                padding: 10px;
                margin-bottom: 10px;
            }
            #send-button {
                width: 100%;
                padding: 10px;
            }
        </style>
    </head>
    <body>
        <h1>KLM-Buddy</h1>
        <div id="chat-container"></div>
        <input type="text" id="user-input" placeholder="Enter your question...">
        <button id="send-button">Send</button>

        <script>
            const chatContainer = document.getElementById('chat-container');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');

            function addMessage(role, content) {
                const messageElement = document.createElement('p');
                messageElement.innerHTML = `<strong>${role}:</strong> ${content}`;
                chatContainer.appendChild(messageElement);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            async function sendMessage() {
                const message = userInput.value.trim();
                if (message) {
                    addMessage('You', message);
                    userInput.value = '';

                    try {
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ message: message }),
                        });

                        const data = await response.json();
                        if (data.response) {
                            addMessage('AI', data.response);
                        } else if (data.error) {
                            addMessage('Error', data.error);
                        }
                    } catch (error) {
                        addMessage('Error', 'An error occurred while sending the message.');
                    }
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
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
