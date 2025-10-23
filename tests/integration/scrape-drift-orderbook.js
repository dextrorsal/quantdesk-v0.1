const puppeteer = require('puppeteer');
const fs = require('fs');

async function scrapeDriftOrderBook() {
  console.log('🚀 Starting Drift order book scraping...');
  
  const browser = await puppeteer.launch({
    headless: false, // Set to true if you don't want to see the browser
    defaultViewport: { width: 1920, height: 1080 },
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  try {
    const page = await browser.newPage();
    
    // Set user agent to avoid detection
    await page.setUserAgent('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
    
    console.log('📡 Navigating to Drift...');
    await page.goto('https://dapp.drift.trade/BTC-PERP', {
      waitUntil: 'networkidle2',
      timeout: 30000
    });

    // Wait for the page to load completely
    console.log('⏳ Waiting for page to load...');
    await page.waitForTimeout(5000);

    // Try to find the order book element with multiple selectors
    console.log('🔍 Looking for order book element...');
    
    const orderBookSelectors = [
      '[data-testid="orderbook"]',
      '[data-testid="order-book"]',
      '.orderbook',
      '.order-book',
      '[class*="orderbook"]',
      '[class*="order-book"]',
      '[class*="OrderBook"]',
      'div[class*="orderbook"]',
      'div[class*="order-book"]'
    ];

    let orderBookElement = null;
    let usedSelector = null;

    for (const selector of orderBookSelectors) {
      try {
        await page.waitForSelector(selector, { timeout: 2000 });
        orderBookElement = await page.$(selector);
        if (orderBookElement) {
          usedSelector = selector;
          console.log(`✅ Found order book with selector: ${selector}`);
          break;
        }
      } catch (e) {
        // Continue to next selector
      }
    }

    if (!orderBookElement) {
      console.log('❌ Order book element not found with common selectors. Let me try a different approach...');
      
      // Take a screenshot to see what's on the page
      await page.screenshot({ path: 'drift-page.png', fullPage: true });
      console.log('📸 Screenshot saved as drift-page.png');
      
      // Get all div elements and look for ones that might be order books
      const allDivs = await page.evaluate(() => {
        const divs = Array.from(document.querySelectorAll('div'));
        return divs.map(div => ({
          className: div.className,
          id: div.id,
          textContent: div.textContent?.substring(0, 100),
          hasPriceData: div.textContent?.includes('$') || div.textContent?.includes('Price'),
          hasOrderData: div.textContent?.includes('Buy') || div.textContent?.includes('Sell') || div.textContent?.includes('Order')
        })).filter(div => div.hasPriceData || div.hasOrderData);
      });

      console.log('🔍 Found potential order book elements:');
      allDivs.forEach((div, index) => {
        console.log(`${index + 1}. Class: ${div.className}, Text: ${div.textContent}`);
      });
    }

    if (orderBookElement) {
      console.log('📊 Extracting order book data...');
      
      // Extract the order book data
      const orderBookData = await page.evaluate((selector) => {
        const element = document.querySelector(selector);
        if (!element) return null;

        // Get computed styles
        const styles = window.getComputedStyle(element);
        const styleObj = {};
        for (let i = 0; i < styles.length; i++) {
          const prop = styles[i];
          styleObj[prop] = styles.getPropertyValue(prop);
        }

        return {
          html: element.outerHTML,
          innerHTML: element.innerHTML,
          styles: styleObj,
          boundingRect: element.getBoundingClientRect(),
          className: element.className,
          id: element.id,
          textContent: element.textContent
        };
      }, usedSelector);

      // Save the data
      fs.writeFileSync('drift-orderbook-data.json', JSON.stringify(orderBookData, null, 2));
      console.log('💾 Order book data saved to drift-orderbook-data.json');

      // Take a screenshot of just the order book
      await orderBookElement.screenshot({ path: 'drift-orderbook.png' });
      console.log('📸 Order book screenshot saved as drift-orderbook.png');

      // Extract CSS classes and try to find the stylesheet
      const cssData = await page.evaluate(() => {
        const stylesheets = Array.from(document.styleSheets);
        const cssRules = [];
        
        stylesheets.forEach(sheet => {
          try {
            const rules = Array.from(sheet.cssRules || sheet.rules || []);
            rules.forEach(rule => {
              if (rule.selectorText && rule.selectorText.includes('orderbook')) {
                cssRules.push({
                  selector: rule.selectorText,
                  styles: rule.style.cssText
                });
              }
            });
          } catch (e) {
            // Cross-origin stylesheets might throw errors
          }
        });
        
        return cssRules;
      });

      fs.writeFileSync('drift-orderbook-css.json', JSON.stringify(cssData, null, 2));
      console.log('🎨 CSS rules saved to drift-orderbook-css.json');

    } else {
      console.log('❌ Could not find order book element. Please check the screenshot and selectors.');
    }

  } catch (error) {
    console.error('❌ Error during scraping:', error);
  } finally {
    await browser.close();
    console.log('🔚 Browser closed. Scraping complete!');
  }
}

// Run the scraper
scrapeDriftOrderBook().catch(console.error);
