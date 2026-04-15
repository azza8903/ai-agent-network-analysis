# Use an official, lightweight Python runtime
FROM python:3.11-slim

# Ensure output is not buffered (helpful for logs)
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Agent script to use. Override at build time with:
#   --build-arg AGENT_FILE=planner_executor_agent.py
ARG AGENT_FILE=react_agent.py

# Copy project files from build context root
# Build this image from the repository root like:
#   docker build -t react-agent .
#   docker build -t planner-agent --build-arg AGENT_FILE=planner_executor_agent.py .
COPY ./${AGENT_FILE} /app/
COPY ./requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Make sure the main script is executable
RUN test -f /app/${AGENT_FILE} || (echo "Error: /app/${AGENT_FILE} not found" && false)
RUN chmod +x /app/${AGENT_FILE} || true

# Persist the build arg so the entrypoint can reference it at runtime
ENV AGENT_FILE=${AGENT_FILE}

# Run the agent as the container entrypoint so CLI args are forwarded:
# Example run:
#   docker run --rm react-agent --prompt "Explain Docker to me"
ENTRYPOINT ["sh", "-c", "exec python -u /app/$AGENT_FILE \"$@\"", "--"]
CMD ["--help"]