import os
import glob
import platform
import subprocess
import time
from datetime import datetime
import psutil
import pyautogui
import pygetwindow as gw
from pygetwindow import PyGetWindowException
from pydantic import ValidationError
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# same local model as Lab 5, temperature 0 so it picks tools the same way every run
llm = ChatOllama(model="llama3.2:3b", temperature=0)


# Calculator and Notepad have to be the active window before pyautogui can type into them
def focus_window(title):
    windows = gw.getWindowsWithTitle(title)
    if windows:
        try:
            windows[0].activate()
        except PyGetWindowException:
            pass  # activate() sometimes throws "error code 0" even though it worked
        time.sleep(1)


# the docstring under each tool is what the model reads to decide when to use it
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
    time.sleep(2)  # give Calculator time to open before typing
    focus_window("Calculator")
    pyautogui.write(expression, interval=0.1)
    pyautogui.press("enter")
    return str(eval(expression))


@tool
def get_date_time() -> str:
    """Get the current date and time. Use it when the user asks what the date, day or time is."""
    # gives e.g. Thursday, 26 November 2026, 07:24 PM
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


@tool
def open_notepad(text: str) -> str:
    """Open Windows Notepad and type the given text into it."""
    print("\n[EXECUTING] Opening Notepad...")
    subprocess.Popen("notepad.exe")
    time.sleep(2)
    focus_window("Notepad")
    pyautogui.write(text, interval=0.05)
    return "Opened Notepad and wrote: " + text


@tool
def search_files(folder: str, pattern: str) -> str:
    """Search a folder and its subfolders for files by name or extension.
    pattern can be an extension like "pdf" or a name like "report"."""
    if not os.path.isdir(folder):
        return "Folder not found: " + folder

    # the model sends "pdf", ".pdf" or "*.pdf", so turn them all into one glob pattern
    extensions = ["pdf", "txt", "doc", "docx", "xlsx", "pptx", "csv", "py", "png", "jpg", "md", "json"]
    if "*" not in pattern:
        if pattern.startswith("."):
            pattern = "*" + pattern
        elif pattern.lower() in extensions:
            pattern = "*." + pattern
        else:
            pattern = "*" + pattern + "*"

    # "**" with recursive=True also looks inside the subfolders
    matches = glob.glob(os.path.join(folder, "**", pattern), recursive=True)
    if not matches:
        return f"No files matching {pattern} in {folder}"
    # total on its own line: with a long list the 3B model read "123" out of a file name
    # and said 123 files when there were 445, so it also only sees the first 10 names now
    return f"Total: {len(matches)} file(s)\n" + "\n".join(matches[:10])


@tool
def create_file(folder: str, file_name: str, content: str) -> str:
    """Create a new text file in a folder and write the given content into it."""
    if not os.path.isdir(folder):
        return "Folder not found: " + folder
    path = os.path.join(folder, file_name)
    # don't let the model overwrite a file that's already there
    if os.path.exists(path):
        return path + " already exists, so I didn't overwrite it."
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return "Created " + path


@tool
def get_system_info() -> str:
    """Get this computer's specifications: operating system, processor and memory (RAM)."""
    memory = psutil.virtual_memory()
    # psutil gives memory in bytes, so divide by 1024**3 for GB
    return (f"Operating system: {platform.system()} {platform.release()} (version {platform.version()})\n"
            f"Processor: {platform.processor()}\n"
            f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical\n"
            f"Memory: {memory.total / 1024**3:.1f} GB total, {memory.available / 1024**3:.1f} GB free")


# register all six tools with the model; bind_tools sends their names and docstrings to it
tools = [open_calculator, get_date_time, open_notepad, search_files, create_file, get_system_info]
llm_with_tools = llm.bind_tools(tools)

# the model answers with a tool's name, so this finds the actual function by that name
tools_by_name = {t.name: t for t in tools}

# a 3B model picks the right tool more often when the tools are listed in the prompt too
system_prompt = """You are a helpful assistant on a Windows computer. You have these tools:
- open_calculator: any calculation
- get_date_time: today's date or the time
- open_notepad: open Notepad and write text
- search_files: find files in a folder
- create_file: make a new text file
- get_system_info: this computer's specifications
Always use a tool when one fits the request. Copy Windows paths exactly as the user wrote them.
After a tool answers, reply to the user in one or two short sentences.
If the tool result has a number in it, use that exact number."""

print("Multi-tool agent ready. Type exit to stop.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break

    # no memory yet: every question starts a new conversation (memory comes in part B)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
    response = llm_with_tools.invoke(messages)

    # the model either answers straight away or asks for one or more tools
    if response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            print("\n[TOOL SELECTED]", tool_call["name"])
            print("[TOOL ARGUMENTS]", tool_call["args"])

            selected = tools_by_name.get(tool_call["name"])
            if selected is None:
                # the small model sometimes invents a tool name
                result = "There is no tool called " + tool_call["name"]
            else:
                try:
                    result = selected.invoke(tool_call["args"])
                except ValidationError:
                    result = "The tool was called with the wrong arguments."

            print("[TOOL RESULT]", result)
            # the result goes back to the model, matched to its request by the call id
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        # second call without tools, so the model just writes the answer
        response = llm.invoke(messages)

    print("\nAgent:", response.content)
