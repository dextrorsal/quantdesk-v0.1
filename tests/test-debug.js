const { chromium } = require('playwright');

async function testPageWithConsole() {
  const browser = await chromium.launch({ 
    headless: false,
    slowMo: 500
  });
  
  const page = await browser.newPage();
  
  // Capture console logs
  const consoleLogs = [];
  page.on('console', msg => {
    consoleLogs.push({
      type: msg.type(),
      text: msg.text(),
      timestamp: new Date().toISOString()
    });
    console.log(`[${msg.type()}] ${msg.text()}`);
  });
  
  // Capture errors
  page.on('pageerror', error => {
    console.log(`🚨 Page Error: ${error.message}`);
  });
  
  try {
    console.log('🌐 Navigating to QuantDesk...');
    await page.goto('http://localhost:3001/lite');
    
    // Wait for page to load
    await page.waitForTimeout(5000);
    
    // Check for React errors
    const reactErrors = await page.evaluate(() => {
      const errorElements = document.querySelectorAll('[data-react-error]');
      return Array.from(errorElements).map(el => el.textContent);
    });
    
    if (reactErrors.length > 0) {
      console.log('🚨 React Errors found:', reactErrors);
    }
    
    // Check if DebugAccountState component exists
    const debugExists = await page.evaluate(() => {
      // Look for the debug component by its content
      const elements = document.querySelectorAll('*');
      for (let el of elements) {
        if (el.textContent && el.textContent.includes('Debug Account State')) {
          return true;
        }
      }
      return false;
    });
    
    console.log(`🔍 Debug component exists: ${debugExists}`);
    
    // Get all text content
    const allText = await page.evaluate(() => document.body.textContent);
    console.log('📝 All text content:', allText.substring(0, 1000));
    
    // Check for any elements with "debug" in their content
    const debugElements = await page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      const debugEls = [];
      for (let el of elements) {
        if (el.textContent && el.textContent.toLowerCase().includes('debug')) {
          debugEls.push({
            tagName: el.tagName,
            textContent: el.textContent.substring(0, 100),
            className: el.className
          });
        }
      }
      return debugEls;
    });
    
    console.log('🔍 Elements with "debug" in content:', debugElements);
    
    // Take another screenshot
    await page.screenshot({ path: 'quantdesk-debug.png', fullPage: true });
    
    console.log('✅ Test completed! Check quantdesk-debug.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testPageWithConsole();
