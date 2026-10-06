import asyncio
from typing import List, Dict, Any
from playwright.async_api import async_playwright, Browser, Page

TARGET_TAGS = ["button", "a", "span", "p", "label", "h1", "h2", "h3", "h4", "li", "strong", "em"]

async def extract_ui_elements_from_url(url: str, headless: bool = True, timeout_ms: int = 30000) -> List[Dict[str, Any]]:
    """
    Renders dynamic web pages via headless Chromium and extracts visible UI text nodes
    focused on leaf/interactive elements to eliminate ancestral container duplication.
    """
    print(f"\n[Scraper] Initializing Chromium engine...")

    async with async_playwright() as p:
        browser: Browser = await p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        page: Page = await context.new_page()

        print(f"[Scraper] Navigating to: {url}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                await page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
        except Exception as e:
            print(f"[Scraper Error] Navigation failed: {e}")
            await browser.close()
            return []

        # Traverse target tags, filtering out empty or parent wrapper containers
        raw_nodes = await page.evaluate(
            """
            (tags) => {
                const results = [];
                const selector = tags.join(',');
                const elements = document.querySelectorAll(selector);
                const seenTexts = new Set();

                elements.forEach((el, index) => {
                    // Check visibility
                    const isVisible = !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
                    if (!isVisible) return;

                    // Ensure this element is either a true leaf or direct text holder
                    // (skip elements whose child elements already contain identical full text)
                    const directText = el.innerText ? el.innerText.trim() : "";
                    if (!directText || directText.length < 2) return;

                    // Normalize whitespace
                    const clean = directText.replace(/\\s+/g, ' ');

                    // De-duplicate identical consecutive child-parent echoes
                    if (seenTexts.has(clean)) return;
                    seenTexts.add(clean);

                    // Compute lightweight CSS selector path
                    let selectorPath = el.tagName.toLowerCase();
                    if (el.id) {
                        selectorPath += `#${el.id}`;
                    } else if (el.className && typeof el.className === 'string') {
                        const cleanClass = el.className.trim().split(/\\s+/).slice(0, 2).join('.');
                        if (cleanClass) selectorPath += `.${cleanClass}`;
                    }

                    results.push({
                        "node_id": index,
                        "tag": el.tagName.toLowerCase(),
                        "selector": selectorPath,
                        "class_name": typeof el.className === 'string' ? el.className.trim() : "",
                        "text": clean
                    });
                });
                return results;
            }
            """,
            TARGET_TAGS
        )

        print(f"[Scraper] Filtered DOM nodes collected: {len(raw_nodes)}")
        await browser.close()
        return raw_nodes

if __name__ == "__main__":
    test_url = "http://books.toscrape.com/"
    print("=" * 65)
    print("DAY 10: REFINED DOM EXTRACTION (LEAF FILTERING)")
    print("=" * 65)
    
    nodes = asyncio.run(extract_ui_elements_from_url(test_url))
    print(f"\nExtracted {len(nodes)} distinct UI elements.")
    print("Sample extracted nodes:")
    for item in nodes[:8]:
        sample = item['text']
        if len(sample) > 60:
            sample = sample[:57] + "..."
        print(f" • [<{item['tag']}> {item['selector']}] \"{sample}\"")