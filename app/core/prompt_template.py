from langchain_core.prompts import PromptTemplate

# Prompt for analyzing log messages using Groq LLM
def get_log_analysis_prompt():
    template = """
You are an expert log analyst. Analyze the following log lines and identify:

1. Errors or warnings present.
2. Possible root causes.
3. Suggestions for fixing the issues.
4. Which services/components might be affected.

Format your response in markdown with clear headers.

Log Content:
--------------
{log_lines}
--------------
"""
    return PromptTemplate(
        input_variables=["log_lines"],
        template=template.strip()
    )
