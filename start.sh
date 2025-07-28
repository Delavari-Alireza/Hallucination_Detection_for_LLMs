
until curl -s http://ollama:11434 > /dev/null; do
  echo "Waiting for Ollama to start..."
  sleep 1
done

echo "Ollama is ready. Starting app..."

curl http://ollama:11434/api/pull -d '{"name":"llama3.2:1b"}'



# Start Gradio in the background
python app.py &


# Start FastAPI in the background
uvicorn api:app --host 0.0.0.0 --port 8000 &

echo "✅ Services started:"
echo "   • REST API → http://0.0.0.0:8000/infer"
echo "   • Gradio UI → http://0.0.0.0:7860"

wait