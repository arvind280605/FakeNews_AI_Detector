import requests
import pandas as pd
from datetime import datetime, timezone
import time

# Facebook-style keywords (social/news topics)
fb_keywords = [
    "Elections",
    "Protests",
    "Local News",
    "Weather Updates",
    "Sports Events",
    "Celebrity News",
    "Traffic Updates",
    "Community Programs",
    "Festivals",
    "Health Alerts"
]

limit_per_request = 100  # Max per Reddit request
total_posts_needed = 500

posts_list = []

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

for query in fb_keywords:
    after = None  # Reset pagination for each keyword
    while len(posts_list) < total_posts_needed:
        url = f"https://www.reddit.com/search.json?q={query}&limit={limit_per_request}"
        if after:
            url += f"&after={after}"

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching data for '{query}': {e}")
            break

        children = data.get('data', {}).get('children', [])
        if not children:
            break  # No more posts for this query

        for post in children:
            post_data = post.get('data', {})
            post_id = post_data.get("id", "")
            posts_list.append({
                "User": post_data.get("author", ""),
                "Message": post_data.get("title", ""),
                "Created_Time": datetime.fromtimestamp(
                    post_data.get("created_utc", 0),
                    tz=timezone.utc
                ).strftime('%Y-%m-%d %H:%M:%S'),
                "Comments": post_data.get("num_comments", 0),
                "Reactions": post_data.get("score", 0),
                "Post_URL": f"https://facebook.com/posts/{post_id}"  # Facebook-style URL
            })

            if len(posts_list) >= total_posts_needed:
                break

        after = data.get('data', {}).get('after')
        if not after:
            break  # Reached end of results

        time.sleep(1)  # Be polite to Reddit servers

# Convert to DataFrame
df = pd.DataFrame(posts_list)

# Save CSV
df.to_csv("facebook_style_posts.csv", index=False)
print(f"Scraping completed! Total posts fetched: {len(posts_list)}. Saved as 'facebook_style_posts.csv'")
