import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

TARGET_USER = "google"      # username you want to scrape
OUTPUT_FILE = "instagram_posts.csv"
POST_LIMIT = 30             # number of posts to scrape

# --------------------------
# Selenium setup
# --------------------------
options = Options()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--log-level=3")
options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(options=options)

url = f"https://www.instagram.com/{TARGET_USER}/"
driver.get(url)
time.sleep(5)   # wait for JS to load

# scroll to load posts
scrolls = 5
for i in range(scrolls):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)

soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

# --------------------------
# Extract post links
# --------------------------
posts = soup.find_all("a", href=True)
links = []

for p in posts:
    href = p["href"]
    if "/p/" in href or "/reel/" in href:
        links.append("https://www.instagram.com" + href)

links = list(dict.fromkeys(links))   # remove duplicates
links = links[:POST_LIMIT]

print(f"Found {len(links)} posts.")

# --------------------------
# Visit each post and extract caption
# --------------------------
data = []

driver = webdriver.Chrome(options=options)

for link in links:
    driver.get(link)
    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # caption selector
    caption_tag = soup.find("h1")

    caption = caption_tag.text.strip() if caption_tag else ""

    data.append([link, caption])
    print("✔ Saved:", caption[:50])

driver.quit()

# --------------------------
# Save CSV
# --------------------------
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["url", "caption"])
    writer.writerows(data)

print(f"\n🎉 DONE! Saved {len(data)} posts → {OUTPUT_FILE}")
