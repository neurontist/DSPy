import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM(model='gemini/gemini-2.5-flash', max_tokens=3000)
dspy.settings.configure(lm=lm)

class ToolSignature(dspy.Signature):
    """Signature for manual tool handling."""
    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    outputs: dspy.ToolCalls = dspy.OutputField()

def weather(city: str) -> str:
    """Get weather information for a city."""
    return f"The weather in {city} is sunny"

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # Note: Use safely in production
        return f"The result is {result}"
    except:
        return "Invalid expression"

# Create tool instances
tools = {
    "weather": dspy.Tool(weather),
    "calculator": dspy.Tool(calculator)
}

# Create predictor
predictor = dspy.Predict(ToolSignature)

# Make a prediction
response = predictor(
    question="What's the weather in New York?",
    tools=list(tools.values())
)

# Execute the tool calls
for call in response.outputs.tool_calls:

    result = tools[call.name](**call.args)
    print(f"Tool: {call.name}")
    print(f"Args: {call.args}")
    print(f"Result: {result}")
