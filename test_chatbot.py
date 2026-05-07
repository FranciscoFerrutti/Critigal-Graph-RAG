#!/usr/bin/env python
"""
Script de test para Critical Graph RAG Chatbot.

Verifica:
1. Carga correcta del agente.
2. Funcionamiento básico del endpoint /chat.
3. Health check del servidor.

Uso:
    python test_chatbot.py              # Test local
    python test_chatbot.py --url <url>  # Test remoto (Render, etc.)
"""

import argparse
import json
import logging
import sys
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def test_health(base_url: str) -> bool:
    """Test: Health check del servidor."""
    logger.info(f"🏥 Health check: GET {base_url}/health")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        logger.info(f"   ✓ Status: {data.get('status')}")
        logger.info(f"   ✓ Agent ready: {data.get('agent_ready')}")
        return data.get("agent_ready", False)
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return False


def test_chat_es(base_url: str, top_k: Optional[int] = None) -> bool:
    """Test: Chat en español."""
    logger.info(f"💬 Chat test (ES): POST {base_url}/chat")
    
    payload = {"message": "¿Cuáles fueron los eventos principales en Gaza?"}
    if top_k:
        payload["top_k"] = top_k
    
    logger.info(f"   Payload: {json.dumps(payload)}")
    
    try:
        response = requests.post(
            f"{base_url}/chat",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"   ✓ Language detected: {data.get('language')}")
        logger.info(f"   ✓ Response (first 100 chars):")
        resp_text = data.get("response", "")[:100]
        logger.info(f"      {resp_text}...")
        
        return True
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                err_detail = e.response.json()
                logger.error(f"      Detail: {err_detail}")
            except:
                logger.error(f"      Response: {e.response.text}")
        return False


def test_chat_en(base_url: str) -> bool:
    """Test: Chat en inglés."""
    logger.info(f"💬 Chat test (EN): POST {base_url}/chat")
    
    payload = {"message": "What events happened in Gaza in 2023?"}
    logger.info(f"   Payload: {json.dumps(payload)}")
    
    try:
        response = requests.post(
            f"{base_url}/chat",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"   ✓ Language detected: {data.get('language')}")
        logger.info(f"   ✓ Response (first 100 chars):")
        resp_text = data.get("response", "")[:100]
        logger.info(f"      {resp_text}...")
        
        return True
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return False


def test_empty_message(base_url: str) -> bool:
    """Test: Verificar que se rechacen mensajes vacíos."""
    logger.info(f"⚠️ Error handling test: Empty message")
    
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={"message": ""},
            timeout=10,
        )
        
        # Debe devolver error
        if response.status_code != 200:
            logger.info(f"   ✓ Correctly rejected with status {response.status_code}")
            return True
        else:
            logger.error(f"   ✗ Should have rejected empty message")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"   ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Critical Graph RAG Chatbot")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL del servidor (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k para retrieval (opcional)",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Saltarse tests de chat (solo health check)",
    )
    
    args = parser.parse_args()
    
    base_url = args.url.rstrip("/")
    logger.info(f"🧪 Testing chatbot at: {base_url}\n")
    
    results = {}
    
    # Test 1: Health check
    logger.info("=" * 60)
    logger.info("TEST 1: Health Check")
    logger.info("=" * 60)
    results["health"] = test_health(base_url)
    
    if not results["health"]:
        logger.error("\n❌ Server not ready. Aborting remaining tests.")
        return 1
    
    if args.skip_chat:
        logger.info("\n⏭️ Skipping chat tests (--skip-chat)")
        print_summary(results)
        return 0
    
    # Test 2: Chat en español
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Chat in Spanish")
    logger.info("=" * 60)
    results["chat_es"] = test_chat_es(base_url, args.top_k)
    
    # Test 3: Chat en inglés
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Chat in English")
    logger.info("=" * 60)
    results["chat_en"] = test_chat_en(base_url)
    
    # Test 4: Error handling
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Error Handling")
    logger.info("=" * 60)
    results["error_handling"] = test_empty_message(base_url)
    
    print_summary(results)
    
    # Retornar 0 si todos los tests pasaron, 1 si alguno falló
    return 0 if all(results.values()) else 1


def print_summary(results: dict[str, bool]):
    """Imprime resumen de resultados."""
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    logger.info(f"\n📈 {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed!")
    else:
        logger.warning("\n⚠️ Some tests failed. Check logs above.")


if __name__ == "__main__":
    sys.exit(main())
