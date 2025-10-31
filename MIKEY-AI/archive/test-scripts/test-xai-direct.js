// test-xai-direct.js
require('dotenv').config({ path: './.env' });

async function testXAIDirect() {
  console.log('🧪 Testing XAI (Grok) Official SDK...');
  
  try {
    const OpenAI = require('openai');
    const client = new OpenAI({
      apiKey: process.env.XAI_API_KEY,
      baseURL: 'https://api.x.ai/v1',
    });

    console.log('📡 Calling XAI API...');
    const response = await client.chat.completions.create({
      model: 'grok-4', // Official model name from XAI docs
      messages: [{ role: 'user', content: 'Hello! Test message.' }],
      temperature: 0.7,
    });

    console.log('✅ XAI Response:', response.choices[0].message.content);
    
  } catch (error) {
    console.log('❌ XAI Error:', error.message);
    
    // Try alternative model names
    if (error.message.includes('model') || error.message.includes('not found')) {
      console.log('\n🔄 Trying alternative model names...');
      
      const alternativeModels = ['grok-2', 'grok-beta', 'grok'];
      
      for (const model of alternativeModels) {
        try {
          console.log(`Trying model: ${model}`);
          const OpenAI = require('openai');
          const client = new OpenAI({
            apiKey: process.env.XAI_API_KEY,
            baseURL: 'https://api.x.ai/v1',
          });

          const response = await client.chat.completions.create({
            model: model,
            messages: [{ role: 'user', content: 'Hello! Test message.' }],
            temperature: 0.7,
          });

          console.log(`✅ XAI Response with ${model}:`, response.choices[0].message.content);
          break;
        } catch (modelError) {
          console.log(`❌ ${model} failed:`, modelError.message);
        }
      }
    }
  }
}

testXAIDirect();
