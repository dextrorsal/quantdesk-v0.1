// test-google-direct.js
require('dotenv').config({ path: './.env' });

async function testGoogleDirect() {
  console.log('🧪 Testing Google Gemini Official SDK...');
  
  try {
    const { GoogleGenAI } = require('@google/genai');
    const ai = new GoogleGenAI({
      apiKey: process.env.GOOGLE_API_KEY,
    });

    console.log('📡 Calling Google Gemini API...');
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: 'Hello! Test message.',
    });

    console.log('✅ Google Response:', response.text);
    
  } catch (error) {
    console.log('❌ Google Error:', error.message);
  }
}

testGoogleDirect();
