"""
Chatbot assistant for the GeneHack AMR application powered by Google's Gemini models.
"""
import json
import streamlit as st
from typing import List, Dict, Any
import google.generativeai as genai
import os

# ==========================
# Initialize Gemini client
# ==========================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

# Use Gemini model (fast version)
model = genai.GenerativeModel("gemini-1.5-flash")

# ==========================
# System prompt
# ==========================
SYSTEM_PROMPT = """
You are GeneHack Assistant, an AI expert in antimicrobial resistance (AMR) genomics.
You analyze genetic data and provide insights on:
1. Gene functions and their role in antimicrobial resistance
2. Protein structures and their significance
3. Resistance mechanisms for different antibiotics
4. Interpretation of AMR analysis results
5. Potential research directions and clinical implications

Current analysis data will be provided in JSON format. Use this data when answering questions.
Keep your responses scientifically accurate but understandable to researchers and healthcare professionals.
If you're unsure about something, acknowledge the limitations rather than making up information.

For any clinical advice, emphasize that your suggestions are for research purposes only and should be 
validated by proper clinical testing and medical professionals.
"""

# ==========================
# Chat history
# ==========================
def initialize_chat_history() -> List[Dict[str, str]]:
    """
    Initialize a new chat history with just the system message.
    """
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def add_analysis_context(chat_history: List[Dict[str, str]], analysis_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Add the current analysis data to the chat history as context.
    """
    context = json.dumps(analysis_data, indent=2)
    context_message = {
        "role": "system",
        "content": f"Here is the current analysis data:\n```json\n{context}\n```\nUse this information to answer the user's questions."
    }

    if len(chat_history) > 1:
        return [chat_history[0], context_message] + chat_history[1:]
    else:
        return chat_history + [context_message]


# ==========================
# Chat with Gemini
# ==========================
def chat_with_assistant(chat_history: List[Dict[str, str]], user_message: str) -> Dict[str, Any]:
    """
    Send a message to the Gemini assistant and get a response.
    """
    if not GEMINI_API_KEY:
        return {
            "response": "Error: Gemini API key is not configured. Please add your API key.",
            "chat_history": chat_history
        }

    chat_history.append({"role": "user", "content": user_message})

    try:
        # Convert chat history into a text prompt
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

        response = model.generate_content(history_text)
        assistant_message = response.text

        chat_history.append({"role": "assistant", "content": assistant_message})

        return {"response": assistant_message, "chat_history": chat_history}

    except Exception as e:
        return {"response": f"Error communicating with Gemini: {str(e)}", "chat_history": chat_history}


# ==========================
# Suggestions generator
# ==========================
def generate_analysis_suggestions(analysis_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Generate suggested questions and research directions based on the analysis.
    """
    if not GEMINI_API_KEY:
        return {
            "suggested_questions": [
                "What are the most common resistance mechanisms in this sample?",
                "Which antibiotics should be avoided based on this analysis?",
                "What are the key resistance genes identified?",
                "How do these genes confer resistance?"
            ],
            "research_directions": [
                "Investigate the prevalence of these resistance genes in your region",
                "Test alternative antibiotics not affected by these mechanisms",
                "Compare this resistance profile with clinical outcomes"
            ]
        }

    try:
        context = json.dumps(analysis_data, indent=2)
        suggestion_prompt = f"""
        Please analyze the following antimicrobial resistance data and generate:
        1. Five suggested questions a researcher might want to ask about this data
        2. Three potential research directions based on these results

        Data:
        ```json
        {context}
        ```

        Return your response in JSON format:
        {{
            "suggested_questions": ["question1", "question2", ...],
            "research_directions": ["direction1", "direction2", ...]
        }}
        """

        response = model.generate_content(suggestion_prompt)

        try:
            suggestions = json.loads(response.text)
            return suggestions
        except:
            return {"error": "Failed to parse Gemini response", "raw": response.text}

    except Exception as e:
        return {
            "suggested_questions": [
                "What are the most common resistance mechanisms in this sample?",
                "Which antibiotics should be avoided based on this analysis?",
                "What are the key resistance genes identified?",
                "How do these genes confer resistance?"
            ],
            "research_directions": [
                "Investigate the prevalence of these resistance genes in your region",
                "Test alternative antibiotics not affected by these mechanisms",
                "Compare this resistance profile with clinical outcomes"
            ]
        }


# ==========================
# Summary generator
# ==========================
def summarize_key_findings(analysis_data: Dict[str, Any]) -> str:
    """
    Generate a concise summary of the key findings from the analysis.
    """
    if not GEMINI_API_KEY:
        return "Gemini API key is not configured. Please add your API key."

    try:
        context = json.dumps(analysis_data, indent=2)
        summary_prompt = f"""
        Please analyze the following antimicrobial resistance data and create a concise summary (250 words max)
        highlighting the key findings, focusing on:

        1. The most significant resistance genes and mechanisms
        2. Antibiotics that would be most and least effective
        3. Any unusual or noteworthy patterns

        Data:
        ```json
        {context}
        ```
        """

        response = model.generate_content(summary_prompt)
        return response.text

    except Exception as e:
        return f"Error generating summary: {str(e)}"
