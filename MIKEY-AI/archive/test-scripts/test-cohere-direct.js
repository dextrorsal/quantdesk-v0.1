// test-cohere-direct.js
require('dotenv').config({ path: './.env' });

async function testCohereDirect() {
  console.log('🧪 Testing Cohere Official SDK...');
  
  try {
    const { CohereClientV2 } = require('cohere-ai');
    const cohere = new CohereClientV2({
      token: process.env.COHERE_API_KEY,
    });

    console.log('📡 Calling Cohere API...');
    const response = await cohere.chat({
      model: 'command-a-03-2025',
      messages: [
        {
          role: 'user',
          content: 'Hello! Test message.',
        },
      ],
    });

    console.log('✅ Cohere Full Response:', JSON.stringify(response, null, 2));
    const contentArray = (response && response.message && Array.isArray(response.message.content))
      ? response.message.content
      : [];
    const firstTextPart = contentArray.find((part) => part && part.type === 'text');
    const text = firstTextPart && firstTextPart.text ? firstTextPart.text : undefined;
    console.log('✅ Cohere Text:', text);
    
  } catch (error) {
    console.log('❌ Cohere Error:', error.message);
  }
}

testCohereDirect();
