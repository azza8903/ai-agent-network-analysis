# Use an official, lightweight Python runtime
FROM python:3.11-slim

# Ensure output is not buffered (helpful for logs)
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy project files from build context root
# Build this image from the repository root like:
#   docker build -t weather-agent -f docker/Dockerfile .
COPY . /app

# Install dependencies if a requirements.txt exists
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Make sure the main script is executable
RUN test -f /app/react_agent.py || (echo "Error: /app/react_agent.py not found" && false)
RUN chmod +x /app/react_agent.py || true

# Run the agent as the container entrypoint so CLI args are forwarded:
# Example run:
#   docker run --rm weather-agent --prompt "Explain Docker to me"
ENTRYPOINT ["python", "-u", "/app/react_agent.py"]
CMD ["--help"]