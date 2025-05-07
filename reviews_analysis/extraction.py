from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
import json
import asyncio


async def main():
    schema = {
        "name": "Software Reviews",
        "baseSelector": "div.i18n-translation_container",    # Repeated elements
        "fields": [
            {
                "name": "review_name",
                "selector": "h3.h5.fw-bold",
                "type": "text"
            },
            {
                "name": "star_rating",
                "selector": "span.star-rating-component >.d-flex > .ms-1",
                "type": "text"
            },
            {
                "name": "comments",
                "selector": "p > span.fw-bold + span",
                "type": "text"
            },
            {
                "name": "pros",
                "selector": "p.fw-bold:has(svg.icon-plus-circle) + p",
                "type": "text"
            },
            {
                "name": "cons",
                "selector": "p.fw-bold:has(svg.icon-minus-circle) + p",
                "type": "text"
            },
        ]
    }
    extraction_strategy = JsonCssExtractionStrategy(schema, verbose=True)
    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=extraction_strategy,
    )

    async with AsyncWebCrawler(verbose=True) as crawler:
        # Example usage
        url = "https://www.capterra.co.uk/reviews/157048/brandwatch"
        result = await crawler.arun(url,
                                    config=config
                                    )
        if result.success:
            data = json.loads(result.extracted_content)
            print(json.dumps(data, indent=4))
            with open("output.json", "w") as f:
                json.dump(data, f, indent=4)
                f.close()
        else:
            print(f"Error scraping the url")
            

if __name__ == "__main__":
    asyncio.run(main())
