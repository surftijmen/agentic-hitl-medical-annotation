from aria.src.core.aria_agent import ARIAAgent
from aria.src.config.config import ARIAConfig

# Initialize configuration
config = ARIAConfig()

# Create ARIA agent
agent = ARIAAgent(config)

# Process an instance with human-in-the-loop
prediction, reasoning = agent.process_instance(instance_data)