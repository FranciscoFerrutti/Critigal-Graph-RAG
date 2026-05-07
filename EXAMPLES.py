"""
Ejemplos de uso del API del chatbot.

Estos scripts demuestran cómo integrar el chatbot en distintos contextos:
- Desde Python (SDK de requests).
- Desde cURL.
- Desde JavaScript/TypeScript.
"""

# =============================================================================
# PYTHON - Usando requests
# =============================================================================

"""
# 1. Chat simple (sin top_k especificado)

import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "¿Cuáles fueron los eventos principales en Gaza en 2023?"}
)
data = response.json()
print(f"Respuesta: {data['response']}")
print(f"Idioma detectado: {data['language']}")


# 2. Chat con parámetros

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "What were the main events in Gaza?",
        "top_k": 20  # Recuperar más eventos
    }
)
print(response.json()['response'])


# 3. Health check

response = requests.get("http://localhost:8000/health")
health = response.json()
print(f"Servidor: {health['status']}")
print(f"Agente listo: {health['agent_ready']}")


# 4. Manejo de errores

try:
    response = requests.post(
        "http://localhost:8000/chat",
        json={"message": ""}  # Mensaje vacío
    )
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    print(f"Error: {e.response.status_code} - {e.response.json()['detail']}")
"""

# =============================================================================
# CURL - Desde línea de comandos
# =============================================================================

"""
# 1. Health check
curl -X GET http://localhost:8000/health

# 2. Chat en español
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Qué sucedió en Gaza?"}'

# 3. Chat en inglés
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What happened in Gaza?", "top_k": 15}'

# 4. Chat desde archivo
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d @- << 'EOF'
{
  "message": "¿Cuáles fueron los eventos de protesta en Palestina?",
  "top_k": 10
}
EOF
"""

# =============================================================================
# JAVASCRIPT / TypeScript - Desde frontend
# =============================================================================

"""
// 1. Fetch básico

async function askChatbot(message) {
  const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Unknown error');
  }
  
  return response.json();
}

// Uso
try {
  const data = await askChatbot('¿Qué sucedió en Gaza?');
  console.log('Respuesta:', data.response);
  console.log('Idioma:', data.language);
} catch (error) {
  console.error('Error:', error.message);
}


// 2. Con axios (si usas React/Vue)

import axios from 'axios';

const chatbotAPI = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
});

async function askChatbot(message, topK = undefined) {
  const payload = { message };
  if (topK !== undefined) payload.top_k = topK;
  
  const response = await chatbotAPI.post('/chat', payload);
  return response.data; // { response, language }
}

// 3. Hook de React

import { useState } from 'react';

export function useChatbot() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const ask = async (message) => {
    setLoading(true);
    setError(null);
    try {
      const data = await askChatbot(message);
      return data.response;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { ask, loading, error };
}

// Componente React
function ChatComponent() {
  const { ask, loading, error } = useChatbot();
  const [messages, setMessages] = useState([]);

  const handleSendMessage = async (userMessage) => {
    try {
      const response = await ask(userMessage);
      setMessages([
        ...messages,
        { role: 'user', content: userMessage },
        { role: 'assistant', content: response },
      ]);
    } catch (err) {
      console.error('Error:', err);
    }
  };

  return (
    <div>
      {messages.map((msg, i) => (
        <div key={i} className={`message ${msg.role}`}>
          {msg.content}
        </div>
      ))}
      <input
        type="text"
        onKeyPress={(e) => {
          if (e.key === 'Enter') handleSendMessage(e.target.value);
        }}
        disabled={loading}
        placeholder="Escribe tu pregunta..."
      />
    </div>
  );
}
"""

# =============================================================================
# PYTHON - Cliente re-utilizable
# =============================================================================

"""
# Crear un cliente para reutilizar en tu app

from typing import Optional
import requests

class CriticalGraphChatbot:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> dict:
        \"\"\"Verifica si el servidor está disponible.\"\"\"
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def ask(self, message: str, top_k: Optional[int] = None) -> str:
        \"\"\"Envía una pregunta y obtiene la respuesta.\"\"\"
        payload = {"message": message}
        if top_k is not None:
            payload["top_k"] = top_k
        
        response = self.session.post(f"{self.base_url}/chat", json=payload)
        response.raise_for_status()
        return response.json()["response"]
    
    def close(self):
        self.session.close()

# Uso
if __name__ == "__main__":
    bot = CriticalGraphChatbot()
    
    # Health check
    health = bot.health_check()
    print(f"Servidor: {health['status']}, Agente: {health['agent_ready']}")
    
    # Hacer preguntas
    response = bot.ask("¿Qué sucedió en Gaza?", top_k=10)
    print(response)
    
    bot.close()
"""

print(__doc__)
