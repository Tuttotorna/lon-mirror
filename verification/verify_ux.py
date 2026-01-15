from playwright.sync_api import sync_playwright
import os

def test_copy_buttons():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the local HTML file
        file_path = os.path.abspath("io_lab_v2_interactive.html")
        page.goto(f"file://{file_path}")

        # Take a screenshot
        page.screenshot(path="verification/io_lab_screenshot.png")
        print("Screenshot saved to verification/io_lab_screenshot.png")

        browser.close()

if __name__ == "__main__":
    test_copy_buttons()