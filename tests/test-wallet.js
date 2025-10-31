const { chromium } = require('playwright');

async function testWithWalletConnection() {
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
    
    console.log('🔍 Looking for wallet connection button...');
    
    // Look for wallet connection button
    const walletButton = await page.locator('text=Connect Wallet').first();
    const walletButtonCount = await walletButton.count();
    console.log(`🔍 Found ${walletButtonCount} wallet connection buttons`);
    
    if (walletButtonCount > 0) {
      console.log('🖱️ Clicking wallet connection button...');
      await walletButton.click();
      
      // Wait for wallet popup
      await page.waitForTimeout(2000);
      
      // Look for Phantom wallet option
      const phantomOption = await page.locator('text=Phantom').first();
      const phantomCount = await phantomOption.count();
      
      if (phantomCount > 0) {
        console.log('👻 Clicking Phantom wallet...');
        await phantomOption.click();
        await page.waitForTimeout(3000);
      } else {
        console.log('⚠️ Phantom wallet option not found');
      }
    }
    
    // Wait for wallet to connect
    await page.waitForTimeout(5000);
    
    // Check if wallet is connected
    const connectedText = await page.locator('text=Connected').count();
    console.log(`🔍 Wallet connected indicators: ${connectedText}`);
    
    // Look for debug component content
    const debugContent = await page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      for (let el of elements) {
        if (el.textContent && el.textContent.includes('Debug Account State')) {
          return el.textContent;
        }
      }
      return null;
    });
    
    if (debugContent) {
      console.log('✅ Debug component found!');
      console.log('📝 Debug content:', debugContent);
    } else {
      console.log('❌ Debug component not found');
    }
    
    // Look for "Wallet Not Connected" message
    const notConnectedText = await page.locator('text=Wallet Not Connected').count();
    console.log(`🔍 "Wallet Not Connected" messages: ${notConnectedText}`);
    
    // Take screenshot
    await page.screenshot({ path: 'quantdesk-wallet-test.png', fullPage: true });
    
    console.log('✅ Test completed! Check quantdesk-wallet-test.png');
    
  } catch (error) {
    console.error('❌ Error:', error);
  } finally {
    await browser.close();
  }
}

testWithWalletConnection();
