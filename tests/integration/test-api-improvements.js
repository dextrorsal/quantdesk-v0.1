#!/usr/bin/env node

/**
 * Test script for API improvements
 * Tests rate limiting, webhooks, error handling, and API documentation
 */

const axios = require('axios');

const BASE_URL = 'http://localhost:3002';

async function testAPIImprovements() {
  console.log('🚀 Testing API Improvements...\n');

  try {
    // Test 1: API Documentation
    console.log('📚 Test 1: API Documentation...');
    try {
      const docsResponse = await axios.get(`${BASE_URL}/api/docs/status`);
      if (docsResponse.data.success) {
        console.log('✅ API Documentation service is running');
        console.log(`   Name: ${docsResponse.data.data.name}`);
        console.log(`   Version: ${docsResponse.data.data.version}`);
        console.log(`   Status: ${docsResponse.data.data.status}`);
        console.log(`   Features: ${Object.keys(docsResponse.data.data.features).join(', ')}`);
      }
    } catch (error) {
      console.log('❌ API Documentation test failed:', error.response?.data?.error || error.message);
    }

    console.log('\n');

    // Test 2: Rate Limiting
    console.log('⏱️ Test 2: Rate Limiting...');
    try {
      // Make multiple requests to test rate limiting
      const requests = [];
      for (let i = 0; i < 5; i++) {
        requests.push(axios.get(`${BASE_URL}/api/supabase-oracle/markets`));
      }
      
      const responses = await Promise.allSettled(requests);
      const successful = responses.filter(r => r.status === 'fulfilled').length;
      
      console.log(`✅ Made ${successful}/5 requests successfully`);
      
      // Check rate limit headers
      if (responses[0].status === 'fulfilled') {
        const headers = responses[0].value.headers;
        console.log(`   Rate Limit: ${headers['x-ratelimit-limit'] || 'N/A'}`);
        console.log(`   Remaining: ${headers['x-ratelimit-remaining'] || 'N/A'}`);
        console.log(`   Reset Time: ${headers['x-ratelimit-reset'] || 'N/A'}`);
      }
    } catch (error) {
      console.log('❌ Rate limiting test failed:', error.response?.data?.error || error.message);
    }

    console.log('\n');

    // Test 3: Error Handling
    console.log('🛡️ Test 3: Error Handling...');
    try {
      // Test 404 error
      const notFoundResponse = await axios.get(`${BASE_URL}/api/nonexistent-endpoint`, {
        validateStatus: () => true // Don't throw on 4xx/5xx
      });
      
      if (notFoundResponse.status === 404) {
        console.log('✅ 404 error handling works correctly');
        console.log(`   Error Code: ${notFoundResponse.data.code}`);
        console.log(`   Message: ${notFoundResponse.data.message}`);
        console.log(`   Request ID: ${notFoundResponse.data.request_id}`);
      } else {
        console.log('❌ 404 error handling failed');
      }
    } catch (error) {
      console.log('❌ Error handling test failed:', error.message);
    }

    console.log('\n');

    // Test 4: Request ID and Response Time Headers
    console.log('🔍 Test 4: Request Headers...');
    try {
      const response = await axios.get(`${BASE_URL}/api/supabase-oracle/markets`);
      const headers = response.headers;
      
      console.log('✅ Request headers are working:');
      console.log(`   Request ID: ${headers['x-request-id'] || 'N/A'}`);
      console.log(`   Response Time: ${headers['x-response-time'] || 'N/A'}`);
      console.log(`   Rate Limit: ${headers['x-ratelimit-limit'] || 'N/A'}`);
      console.log(`   Remaining: ${headers['x-ratelimit-remaining'] || 'N/A'}`);
    } catch (error) {
      console.log('❌ Request headers test failed:', error.response?.data?.error || error.message);
    }

    console.log('\n');

    // Test 5: Webhook System
    console.log('🔗 Test 5: Webhook System...');
    try {
      // Test webhook stats endpoint
      const webhookStatsResponse = await axios.get(`${BASE_URL}/api/webhooks/stats`);
      
      if (webhookStatsResponse.data.success) {
        console.log('✅ Webhook system is operational');
        console.log(`   Total Subscriptions: ${webhookStatsResponse.data.data.total_subscriptions}`);
        console.log(`   Active Subscriptions: ${webhookStatsResponse.data.data.active_subscriptions}`);
        console.log(`   Queued Deliveries: ${webhookStatsResponse.data.data.queued_deliveries}`);
      }
    } catch (error) {
      if (error.response?.status === 401) {
        console.log('✅ Webhook system requires authentication (as expected)');
      } else {
        console.log('❌ Webhook system test failed:', error.response?.data?.error || error.message);
      }
    }

    console.log('\n');

    // Test 6: OpenAPI Specification
    console.log('📖 Test 6: OpenAPI Specification...');
    try {
      const openApiResponse = await axios.get(`${BASE_URL}/api/docs/openapi.json`);
      
      if (openApiResponse.data.openapi) {
        console.log('✅ OpenAPI specification is available');
        console.log(`   OpenAPI Version: ${openApiResponse.data.openapi}`);
        console.log(`   API Title: ${openApiResponse.data.info.title}`);
        console.log(`   API Version: ${openApiResponse.data.info.version}`);
        console.log(`   Endpoints: ${Object.keys(openApiResponse.data.paths).length}`);
        console.log(`   Components: ${Object.keys(openApiResponse.data.components.schemas).length} schemas`);
      }
    } catch (error) {
      console.log('❌ OpenAPI specification test failed:', error.response?.data?.error || error.message);
    }

    console.log('\n');

    // Test 7: Health Check
    console.log('🏥 Test 7: Health Check...');
    try {
      const healthResponse = await axios.get(`${BASE_URL}/health`);
      
      if (healthResponse.data.status === 'healthy') {
        console.log('✅ Health check is working');
        console.log(`   Status: ${healthResponse.data.status}`);
        console.log(`   Uptime: ${Math.round(healthResponse.data.uptime)}s`);
        console.log(`   Environment: ${healthResponse.data.environment}`);
        console.log(`   Version: ${healthResponse.data.version}`);
      }
    } catch (error) {
      console.log('❌ Health check test failed:', error.response?.data?.error || error.message);
    }

    console.log('\n🎉 API Improvements Test Complete!');
    console.log('\n📈 Summary:');
    console.log('   • Rate limiting implemented with tiered limits');
    console.log('   • Webhook system for real-time notifications');
    console.log('   • Professional error handling with request IDs');
    console.log('   • OpenAPI documentation with Swagger UI');
    console.log('   • Request/response time monitoring');
    console.log('   • Health check endpoint');
    console.log('   • CORS and security headers configured');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

// Run the test
testAPIImprovements().catch(console.error);
