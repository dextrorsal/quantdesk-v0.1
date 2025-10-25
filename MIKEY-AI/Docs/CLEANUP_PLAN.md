# Test Scripts Cleanup Plan

## 🧹 **Scripts to KEEP (Essential):**

### **Core Testing:**
- `test-tool-integration.js` - Main integration test
- `test-updated-endpoints.js` - Endpoint verification
- `test-mikey-ai.js` - Comprehensive Mikey AI test
- `test-quantdesk-endpoints.js` - Backend API testing

### **Individual Provider Tests:**
- `test-openai-direct.js` - OpenAI testing
- `test-google-direct.js` - Google Gemini testing
- `test-cohere-direct.js` - Cohere testing
- `test-xai-direct.js` - XAI testing

### **Utility Scripts:**
- `check-running-services.js` - Service monitoring
- `start-quantdesk-api.js` - API startup helper
- `debug-tool-detection.js` - Tool detection debugging

## 🗑️ **Scripts to REMOVE (Redundant/Outdated):**

### **Redundant Tests:**
- `test-hackathon-ready.js` - Superseded by newer tests
- `test-hackathon-final.js` - Superseded by newer tests
- `test-simple-test.js` - Basic test, not needed
- `test-simple-env-test.js` - Environment test, not needed
- `test-full-system.js` - Too complex, use specific tests
- `test-mikey-integration.js` - Superseded by test-tool-integration.js
- `test-real-data-integration.js` - Superseded by test-tool-integration.js
- `test-api-endpoints.js` - Superseded by test-quantdesk-endpoints.js
- `test-external-sources.js` - Superseded by test-quantdesk-endpoints.js
- `test-postman-endpoints.js` - Superseded by test-updated-endpoints.js
- `quick-endpoint-test.js` - Superseded by test-updated-endpoints.js

### **Hugging Face Tests (Not Working):**
- `test-hf-direct.js` - HF API not working
- `test-hf-langchain.js` - HF LangChain not working
- `test-hf-working.js` - HF not working
- `test-multiple-hf-models.js` - HF models not available
- `test-popular-hf-models.js` - HF models not available
- `test-qwen-model.js` - Qwen model not available
- `test-qwen-vl-space.js` - Qwen space not working
- `archive-hf-tests.js` - Already archived

## 📁 **Organize Remaining Scripts:**

### **Create Test Categories:**
```
tests/
├── core/
│   ├── test-tool-integration.js
│   ├── test-mikey-ai.js
│   └── test-updated-endpoints.js
├── providers/
│   ├── test-openai-direct.js
│   ├── test-google-direct.js
│   ├── test-cohere-direct.js
│   └── test-xai-direct.js
├── backend/
│   └── test-quantdesk-endpoints.js
└── utils/
    ├── check-running-services.js
    ├── start-quantdesk-api.js
    └── debug-tool-detection.js
```

## 🎯 **Cleanup Actions:**
1. Remove redundant test scripts
2. Organize remaining scripts into categories
3. Update documentation to reflect current test structure
4. Create a simple test runner script

## 📋 **Final Test Structure:**
- **Core Tests**: 3 essential integration tests
- **Provider Tests**: 4 individual LLM provider tests
- **Backend Tests**: 1 backend API test
- **Utility Scripts**: 3 helper scripts
- **Total**: 11 focused, working scripts (down from 25+)
