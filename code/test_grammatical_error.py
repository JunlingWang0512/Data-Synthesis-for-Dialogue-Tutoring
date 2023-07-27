import language_tool_python

def compute_grammatical_errors(text: str) -> float:
    # Initialize the LanguageTool object for English
    tool = language_tool_python.LanguageTool('en-US')
    
    # Use the correct() method to correct the text
    matches = tool.check(text)
    
    # Calculate grammatical errors
    grammatical_errors = len(matches)
    
    return grammatical_errors

print(compute_grammatical_errors('What are the capital of France'))