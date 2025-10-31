// test-openai-direct.js
require('dotenv').config({ path: './.env' });

async function testOpenAIDirect() {
  console.log('🧪 Testing OpenAI Official SDK...');
  
  try {
    const OpenAI = require('openai');
    const client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    console.log('📡 Calling OpenAI API...');
    const response = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: 'Hello! Test message.' }],
      temperature: 0.7,
    });

    console.log('✅ OpenAI Response:', response.choices[0].message.content);
    
  } catch (error) {
    console.log('❌ OpenAI Error:', error.message);
  }
}

testOpenAIDirect();
