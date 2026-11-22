# AgentMemOS

## Conversation and memory

Run `python3 -m src.main` to initialize SQLite and `python3 -m src.agents.agent`
to chat. The CLI also initializes the database when it starts.

The LangGraph workflow interprets the user turn, validates and applies events in
SQLite, and replies with the result. If more than one saved vehicle matches an
event, the pending clarification is saved per session and scope.

Set `AGENTMEMOS_EXTRACTION_MODEL` to change the Ollama interpreter. The default is
`llama3.1:8b`. Use `AGENTMEMOS_DB_PATH` with a separate SQLite path to replay
conversations without changing your working database.

Run the regression suite with `python3 -m unittest discover -s tests -v` in the
project's Python environment.
