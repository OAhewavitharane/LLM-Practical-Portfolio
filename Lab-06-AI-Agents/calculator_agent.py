import subprocess
import time
import pyautogui
import pygetwindow as gw
from pygetwindow import PyGetWindowException
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

llm = ChatOllama(model="llama3.2:3b", temperature=0)


@tool
def open_calculator(expression: str) -> str:
    """Open the Windows Calculator, type a maths expression into it and return the answer.
    Use it for any calculation, for example "25*4" or "(10+5)/3"."""
    # the model sometimes writes × or ÷ instead of * and /
    expression = expression.replace("×", "*").replace("÷", "/").replace(" ", "")

    # only numbers and + - * / ( ) . so eval can't run anything else
    if not expression or not all(ch in "0123456789+-*/()." for ch in expression):
        return "I can only calculate numbers with + - * / and brackets."

    print("\n[EXECUTING] Opening Windows Calculator...")
    subprocess.Popen("calc.exe")
    time.sleep(2)

    windows = gw.getWindowsWithTitle("Calculator")
    if windows:
        try:
            windows[0].activate()
        except PyGetWindowException:
            pass  # activate() sometimes throws "error code 0" even though it worked
        time.sleep(1)

    pyautogui.write(expression, interval=0.1)
    pyautogui.press("enter")
    print("[SUCCESS] Expression entered into Calculator.")

    return str(eval(expression))


llm_with_tools = llm.bind_tools([open_calculator])

system_prompt = """You are a helpful assistant on a Windows computer.
If the user asks you to calculate anything, you MUST call the open_calculator tool.
Pass the expression with digits and + - * / only, for example 25*4.
After the tool answers, tell the user the result in one short sentence."""

print("Calculator agent ready. Type exit to stop.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break

    # no memory yet: every question starts a new conversation (memory comes in part B)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            print("\n[TOOL SELECTED]", tool_call["name"])
            print("[TOOL ARGUMENTS]", tool_call["args"])
            result = open_calculator.invoke(tool_call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
        # second call without tools, so the model just answers with the result
        response = llm.invoke(messages)

    print("\nAgent:", response.content)
