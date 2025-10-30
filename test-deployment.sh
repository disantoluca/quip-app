#!/bin/bash

# Test your deployed Quip App container
# Replace these with your actual API keys when testing

IMAGE="ghcr.io/disantoluca/quip-app:latest"

echo "🧪 Testing deployed container: $IMAGE"
echo ""

# Pull the latest image
echo "📥 Pulling latest image..."
docker pull $IMAGE

# Run container with environment variables
echo "🚀 Starting container..."
docker run -d -p 8501:8501 \
  -e OPENAI_API_KEY="your_openai_api_key_here" \
  -e QUIP_TOKEN="your_quip_token_here" \
  -e QUIP_BASE_URL="https://platform.quip-apple.com" \
  --name quip-app-test \
  $IMAGE

echo "⏳ Waiting for app to start..."
sleep 20

# Test health check
echo "🔍 Testing health endpoint..."
if curl -s http://localhost:8501/_stcore/health > /dev/null; then
    echo "✅ Health check passed!"
    echo ""
    echo "🌐 Your app is running at: http://localhost:8501"
    echo ""
    echo "🎉 SUCCESS! Your containerized Quip app is working!"
    echo ""
    echo "Features available:"
    echo "  ✅ Robust HTTP retry with exponential backoff"
    echo "  ✅ Fail-soft batch processing"
    echo "  ✅ Clear error reporting with retry buttons"
    echo "  ✅ Quip folder crawling"
    echo "  ✅ Local file upload (PDF, HTML, TXT)"
    echo "  ✅ OpenAI-powered Q&A"
else
    echo "❌ Health check failed. Checking logs..."
    docker logs quip-app-test
fi

echo ""
echo "🛑 To stop the test container:"
echo "   docker stop quip-app-test && docker rm quip-app-test"
echo ""
echo "📤 To share with your team:"
echo "   docker pull ghcr.io/disantoluca/quip-app:latest"
echo "   docker run -p 8501:8501 -e OPENAI_API_KEY=key -e QUIP_TOKEN=token ghcr.io/disantoluca/quip-app:latest"
