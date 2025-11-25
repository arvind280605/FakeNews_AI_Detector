import requests
import pandas as pd
from datetime import datetime, timezone
import time

# Keywords to fetch more posts (Reddit-style topics)
reddit_keywords = [
    "World News",
    "Education",
    "Science",
    "Technology",
    "Environment",
    "Politics",
    "Gaming",
    "Movies",
    "Books",
    "Travel"
]

limit_per_request = 100  # Max per Reddit request
total_posts_needed = 500

posts_list = []

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

for query in reddit_keywords:  # Fixed: use 'reddit_keywords' here
    after = None  # Reset pagination for each keyword
    while len(posts_list) < total_posts_needed:
        url = f"https://www.reddit.com/search.json?q={query}&limit={limit_per_request}"
        if after:
            url += f"&after={after}"

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raises HTTPError for bad responses
            data = response.json()
        except Exception as e:
            print(f"Error fetching data for '{query}': {e}")
            break

        children = data.get('data', {}).get('children', [])
        if not children:
            break  # No more posts for this query

        for post in children:
            post_data = post.get('data', {})
            posts_list.append({
                "Title": post_data.get("title", ""),
                "Author": post_data.get("author", ""),
                "Subreddit": post_data.get("subreddit", ""),
                "Score": post_data.get("score", 0),
                "Comments": post_data.get("num_comments", 0),
                "URL": "https://reddit.com" + post_data.get("permalink", ""),
                "Created_UTC": datetime.fromtimestamp(
                    post_data.get("created_utc", 0),
                    tz=timezone.utc
                ).strftime('%Y-%m-%d %H:%M:%S')
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
df.to_csv("reddit_posts_500.csv", index=False)
print(f"Scraping completed! Total posts fetched: {len(posts_list)}. Saved as 'reddit_posts_500.csv'")
