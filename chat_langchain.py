"""
chat_chain.py
=============
Turns a list of recognized sign words into a natural sentence + a spoken-style
reply, using Gemini 3.6 Flash via LangChain, with conversation history kept
per session.

Setup:
    pip install langchain langchain-google-genai

    Set your API key before running (get one from Google AI Studio):
        Windows (cmd):   set GOOGLE_API_KEY=your-key-here
        Windows (ps):    $env:GOOGLE_API_KEY="your-key-here"
        Mac/Linux:       export GOOGLE_API_KEY=your-key-here
    or set os.environ["GOOGLE_API_KEY"] below directly (fine for local testing,
    don't commit a real key to git).

Used by live_sentence_demo.py as:
    from chat_chain import get_reply
    reply = get_reply(["help", "doctor"])          # uses the default session
    reply = get_reply(["help", "doctor"], session_id="user1")   # named session

Model name note: "gemini-3.6-flash" is current as of mid/late 2026. Google
renames and retires model strings fairly often — if you get a 404 / not-found
error from the API, check https://ai.google.dev/gemini-api/docs/models for
the current string and update MODEL_NAME below.
"""
import os

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

MODEL_NAME = "gemini-3.6-flash"

# os.environ["GOOGLE_API_KEY"] = "put-your-key-here"   # uncomment if you'd rather set it here

SYSTEM_PROMPT = """You are having a spoken conversation with a Deaf or hard-of-hearing \
person who communicates using ASL. Their signs are converted to words by an unreliable \
sign-language recognition model, so you will receive a rough, sometimes wrong, list of \
words guessed from their signing, in order, not full grammatical sentences.

Do this for every turn:
1. Work out the most likely sentence they meant from the word list, fixing obvious \
   recognition slips using context (grammar, meaning, what makes sense together).
2. Reply naturally to that sentence, in 1 to 3 short sentences, plain text only, no \
   markdown, no bullet points, since your reply will be read aloud by text-to-speech.
3. If the words don't form anything sensible even after your best interpretation, \
   say briefly that you didn't catch that and ask them to repeat it, instead of \
   guessing wildly or inventing meaning that isn't there.

Keep replies short, warm, and conversational, like a real spoken exchange."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.4)

chain = prompt | model

_store: dict[str, BaseChatMessageHistory] = {}   # session_id -> history, kept in memory only


def _get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


chat = RunnableWithMessageHistory(
    chain,
    _get_history,
    input_messages_key="input",
    history_messages_key="history",
)


def get_reply(words: list[str], session_id: str = "default") -> str:
    """words: recognized signs in order, e.g. ["help", "doctor"].
    Returns the model's reply as plain text."""
    text = " ".join(words)
    response = chat.invoke(
        {"input": text},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content


def reset_session(session_id: str = "default") -> None:
    """Clear a session's history, e.g. when starting a new conversation."""
    _store.pop(session_id, None)


if __name__ == "__main__":
    # quick manual test: python chat_chain.py
    print(get_reply(["hello"]))
    print(get_reply(["help", "doctor"]))
    print(get_reply(["thank", "you"]))