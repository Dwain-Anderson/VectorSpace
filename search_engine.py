from concurrent.futures import ThreadPoolExecutor
from vector_model import InvertedIndex, VectorCompare
from crawler import SimpleWebCrawler
from database import Database
from vector_database import VectorizedDatabase

def get_cached_results(query, db, vector_compare):
    cached_queries = db.get_all_cached_queries()
    query_vector = vector_compare.concordance(query)

    for cached_query in cached_queries:
        cached_query_vector = vector_compare.concordance(cached_query)
        similarity = vector_compare.relation(query_vector, cached_query_vector)
        if similarity > 0.8:  # Similarity threshold
            cached_result = db.get_query_result(cached_query)
            return cached_result.split('|')  # Return cached results
    return None

def perform_search(query, db):
    vector_compare = VectorCompare()

    # Check cache first
    db.prune_cache()
    db.prune_documents()

    cached_results = get_cached_results(query, db, vector_compare)
    if cached_results:
        return cached_results

    # Initialize inverted index and load documents
    index = InvertedIndex()
    crawler = SimpleWebCrawler(db)
    crawler.crawl(max_depth=100)  # Using Internet Archive as a seed URL

    documents = crawler.get_documents()
    pagerank = crawler.get_pagerank()

    # Save documents and PageRank scores to DB
    with ThreadPoolExecutor() as executor:
        futures = []
        for url, text in documents.items():
            futures.append(executor.submit(db.save_document, url, text))
        futures.append(executor.submit(db.save_pagerank, pagerank))
        for future in futures:
            future.result()

    # Add documents to index
    for doc_url, doc_info in documents.items():
        index.add_document(doc_url, doc_info)

    # Perform search
    search_results = index.search_with_pagerank(query, pagerank)
    result_docs = [documents[doc_id]['content'] for doc_id in search_results]
    result_str = '|'.join(search_results)  # Save result doc IDs as a string

    # Save result to cache
    if len(result_str) >= 0:
        db.save_query_result(query, result_str)

    return result_docs


def main():
    query = input("Enter your search query: ").strip()
    if not query:
        print("Error: The query cannot be empty.")
        return

    db = VectorizedDatabase()
    # Perform search
    results = perform_search(query, db)
    for doc in results:
        print(doc)

if __name__ == '__main__':
    main()
