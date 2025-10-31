const { chromium } = require('playwright');

async function testSimpleComponent() {
  const browser = await chromium.launch({ 
    headless: false,
    slowMo: 1000
  });
  
  const page = await browser.newPage();
  
  // Capture console logs
  page.on('console', msg => {
    console.log(`[${msg.type()}] ${msg.text()}`);
  });
  
  try {
    console.log('🌐 Navigating to QuantDesk...');
    await page.goto('http://localhost:3001/lite');
    
    // Wait for page to load
    await page.waitForTimeout(3000);
    
    // Look for any element with "Debug" text
    const debugElements = await page.locator('text=Debug').count();
    console.log(`🔍 Elements with "Debug" text: ${debugElements}`);
    
    // Look for any element with "Wallet Not Connected" text
    const notConnectedElements = await page.locator('text=Wallet Not Connected').count();
    console.log(`🔍 Elements with "Wallet Not Connected" text: ${notConnectedElements}`);
    
    // Look for any element with "Debug Account State" text
    const debugAccountElements = await page.locator('text=Debug Account State').count();
    console.log(`🔍 Elements with "Debug Account State" text: ${debugAccountElements}`);
    
    // Get all text content and look for debug-related content
    const allText = await page.evaluate(() => document.body.textContent);
    
    // Check if debug component is in the DOM but not visible
    const debugInDOM = await page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      for (let el of elements) {
        if (el.textContent && el.textContent.includes('Debug Account State')) {
          return {
            tagName: el.tagName,
            className: el.className,
            textContent: el.textContent.substring(0, 200),
            style: el.style.cssText,
            hidden: el.style.display === 'none' || el.style.visibility === 'hidden'
          };
        }
      }
      return null;
    });
    
    if (debugInDOM) {
      console.log('✅ Debug component found in DOM:', debugInDOM);
    } else {
      console.log('❌ Debug component not found in DOM');
    }
    
    // Check for any React errors
    const reactErrors = await page.evaluate(() => {
      const errorElements = document.querySelectorAll('[data-react-error]');
      return Array.from(errorElements).map(el => el.textContent);
    });
    
    if (reactErrors.length > 0) {
      console.log('🚨 React Errors found:', reactErrors);
    }
    
    // Take screenshot
    await page.screenshot({ path: 'quantdesk-simple-test.png', fullPage: true });
    
    console.log('✅ Test completed! Check quantdesk-simple-test.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testSimpleComponent();
