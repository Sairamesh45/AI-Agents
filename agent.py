
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
import json
from datetime import datetime


load_dotenv()

OPEN_ROUTER_API = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL") or os.getenv("MODEL") or "google/gemini-2.5-flash"

if not OPEN_ROUTER_API:
    raise ValueError("API key not found. Please set OPENROUTER_API_KEY in .env file.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_API,
)



def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")



def calculator(a, b, operation):
    if(operation == "add"):
        return a+b

    elif(operation == "divide"):
        if(b != 0):
            return a/b
        else: return "Cannot divide with Zero"

    elif(operation == "subtract"):
        return a-b
    
    elif(operation == "multiply"):
        return a*b
    
    else:
        return "Unknown Operation"



class Tool:
    def __init__(self, function, description, parameters):
        self.function = function
        self.description = description
        self.parameters = parameters

    def execute(self, arguments):
        return self.function(**arguments)

    def schema(self):
        return {
            "type" : "function",
            "function":{
                "name": self.function.__name__,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    


calculator_parameters = {
    "type": "object",
    "properties": {
        "a": {
            "type": "number",
            "description": "First number"
        },
        "b": {
            "type": "number",
            "description": "Second number"
        },
        "operation": {
            "type": "string",
            "enum": [
                "add",
                "subtract",
                "multiply",
                "divide"
            ],
            "description": "Mathematical operation"
        }
    },
    "required": ["a", "b", "operation"]
}




calculator_tool = Tool(
    function=calculator,
    description="Perform mathematical calculations",
    parameters=calculator_parameters
)




time_parameters = {
    "type": "object",
    "properties": {}
}



time_tool = Tool(
    function = get_current_time,
    description="Get current Time",
    parameters=time_parameters
)



tool_list = [
    calculator_tool,
    time_tool
]


TOOL_MAP = {
    tool.function.__name__: tool
    for tool in tool_list
}



TOOLS = [
    tool.schema()
    for tool in tool_list
]



messages = []


print(
    f"Agent initialized using model '{MODEL}'. "
    "Type 'exit' to quit.\n"
)

while True:
    try:
        user_input = input("You: ")
    except (EOFError, KeyboardInterrupt):
        break

    if not user_input.strip():
        continue

    if user_input.lower().strip() == "exit":
        break

    messages.append({"role": "user", "content": user_input})

    while True:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                max_tokens=1000
            )

            response_message = response.choices[0].message
            messages.append(response_message)

            if not response_message.tool_calls:
                print("AI: ",response_message.content)
                break

            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name

                tool = TOOL_MAP.get(tool_name)

                if tool:
                    print("The tool is running...")
                    arguments = json.loads(
                    tool_call.function.arguments
                )

                    result = tool.execute(arguments)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps(result)
                    })
                    
                        

        except Exception as e:
            print(f"Error: {e}")
            break


# In[ ]:




