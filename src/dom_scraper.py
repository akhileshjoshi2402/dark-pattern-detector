import asyncio
from typing import List, Dict, Any
from playwright.async_api import async_playwright, Browser, Page

TARGET_TAGS = [
    "button", "a", "span", "p", "label", 
    "h1", "h2", "h3", "h4", "li", "strong", "em", "dialog"
]

async def trigger_scroll_and_lazy_load(page: Page, scroll_delay_ms: int = 400) -> None:
    """
    Simulates human scrolling down the document to trigger lazy-loaded
    DOM nodes, hydration events, and scroll-depth popups.
    """
    total_height = await page.evaluate("() => document.body.scrollHeight")
    viewport_height = 800
    current_position = 0

    while current_position < total_height:
        current_position += viewport_height
        await page.evaluate(f"window.scrollTo(0, {current_position})")
        await page.wait_for_timeout(scroll_delay_ms)
        # Re-evaluate scroll height in case infinite scroll or lazy sections expanded it
        total_height = await page.evaluate("() => document.body.scrollHeight")

    # Scroll back to the top to capture any sticky headers or top-level popups
    await page.evaluate("window.scrollTo(0, 0)")
    await page.wait_for_timeout(500)

async def extract_ui_elements_from_url(
    url: str, 
    headless: bool = True, 
    timeout_ms: int = 30000,
    trigger_events: bool = True
) -> List[Dict[str, Any]]:
    """
    Renders web pages via headless Chromium, executes scrolling and interaction triggers,
    and extracts visible text nodes along with structural and overlay metadata.
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

        if trigger_events:
            print(f"[Scraper] Executing dynamic scroll triggers and lazy-load routines...")
            await trigger_scroll_and_lazy_load(page)

        # In-browser DOM extraction with leaf filtering and modal/overlay detection
        raw_nodes = await page.evaluate(
            """
            (tags) => {
                const results = [];
                const selector = tags.join(',');
                const elements = document.querySelectorAll(selector);
                const seenTexts = new Set();

                elements.forEach((el, index) => {
                    const isVisible = !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
                    if (!isVisible) return;

                    const directText = el.innerText ? el.innerText.trim() : "";
                    if (!directText || directText.length < 2) return;

                    const clean = directText.replace(/\\s+/g, ' ');
                    if (seenTexts.has(clean)) return;
                    seenTexts.add(clean);

                    // Overlay / modal heuristic detection
                    const isOverlay = !!(
                        el.closest('dialog') || 
                        el.closest('[role="dialog"]') || 
                        el.closest('[aria-modal="true"]') ||
                        el.closest('.modal') || 
                        el.closest('.popup') ||
                        el.closest('.overlay')
                    );

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
                        "is_overlay": isOverlay,
                        "text": clean
                    });
                });
                return results;
            }
            """,
            TARGET_TAGS
        )

        print(f"[Scraper] Total extracted UI elements: {len(raw_nodes)}")
        await browser.close()
        return raw_nodes

if __name__ == "__main__":
    test_url = "http://books.toscrape.com/"
    print("=" * 65)
    print("DAY 11: DYNAMIC SCROLL & OVERLAY INTERCEPTION TEST")
    print("=" * 65)
    
    nodes = asyncio.run(extract_ui_elements_from_url(test_url, trigger_events=True))
    print(f"\nExtracted {len(nodes)} distinct UI elements with dynamic triggers.")
    print("Sample extracted nodes:")
    for item in nodes[:6]:
        sample = item['text']
        if len(sample) > 55:
            sample = sample[:52] + "..."
        overlay_flag = "[OVERLAY] " if item['is_overlay'] else ""
        print(f" • {overlay_flag}[<{item['tag']}> {item['selector']}] \"{sample}\"")