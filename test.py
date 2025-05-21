from smolagents import CodeAgent, LiteLLMModel

# Initialize the model with simplified configuration
model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5-coder:3b",
    api_base="http://localhost:11434",
    num_ctx=3000,
    # mode="completion"
)

# Initialize the agent with no tools and no base tools
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

# Run a simple task
agent.run("who is prime minister of India?")