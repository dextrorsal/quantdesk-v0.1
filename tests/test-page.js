const { chromium } = require('playwright');

async function testPage() {
  const browser = await chromium.launch({ 
    headless: false, // Show browser window
    slowMo: 1000 // Slow down for visibility
  });
  
  const page = await browser.newPage();
  
  try {
    console.log('🌐 Navigating to QuantDesk...');
    await page.goto('http://localhost:3001/lite');
    
    // Wait for page to load
    await page.waitForTimeout(3000);
    
    console.log('📸 Taking screenshot...');
    await page.screenshot({ path: 'quantdesk-page.png', fullPage: true });
    
    // Check if debug component exists
    const debugComponent = await page.locator('[class*="DebugAccountState"]').count();
    console.log(`🔍 Debug component found: ${debugComponent} instances`);
    
    // Check for any elements with "debug" in class name
    const debugElements = await page.locator('[class*="debug"], [class*="Debug"]').count();
    console.log(`🔍 Elements with "debug" in class: ${debugElements}`);
    
    // Check for wallet connection button
    const walletButton = await page.locator('[data-wallet-adapter-button]').count();
    console.log(`🔍 Wallet button found: ${walletButton} instances`);
    
    // Check page title
    const title = await page.title();
    console.log(`📄 Page title: ${title}`);
    
    // Get all text content to see what's actually on the page
    const bodyText = await page.locator('body').textContent();
    console.log('📝 Page content preview:', bodyText.substring(0, 500));
    
    // Look for any error messages
    const errors = await page.locator('[class*="error"], [class*="Error"]').count();
    console.log(`❌ Error elements found: ${errors}`);
    
    // Check console logs
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.log('🚨 Console Error:', msg.text());
      }
    });
    
    // Wait a bit more to see any dynamic content
    await page.waitForTimeout(2000);
    
    console.log('✅ Test completed! Check quantdesk-page.png for screenshot');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testPage();
