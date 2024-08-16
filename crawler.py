import concurrent.futures
import threading
import queue
import time
import random
import math
from math import pow, exp, log, log2, floor, ceil
import os
from vector_database import VectorizedDatabase
from statistics import Statistics
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import networkx as nx
from signal import signal, SIGINT
import re
from vector_model import TextProcessor
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class SimpleWebCrawler:
    def __init__(self, db):
        self.db = db
        self.visited = set()
        self.graph = nx.DiGraph()
        self.lock = threading.Lock()
        self.url_queue = queue.Queue()
        self.max_pages = 0
        self.links_seen = set()
        self.num_threads = max(10, 2 * os.cpu_count())
        self.stop_event = threading.Event()
        self.text_processor = TextProcessor()

        # Setup Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                                       options=chrome_options)

    def get_seed(self, url=None):
        if url is None:
            urls = self.db.get_k_urls(k=20)
            if urls is None or urls == []:
                url = 'https://en.wikipedia.org/wiki/Main_Page'
            else:
                N = len(urls) - 1
                i = random.randint(a=0, b=(N // 2 + 1))
                url = urls[i][0]
        return url

    def auto_crawl(self, limit=100, max_depth=2, seed_url=None):
        seed_url = self.get_seed(seed_url) if seed_url is None else seed_url
        if not seed_url:
            print("No seed URL found. Cannot start crawling.")
            return

        self.max_pages = limit
        self.url_queue.put((seed_url, 0))

        # Initialize and start a new statistics session
        stats = Statistics()
        stats.start_new_session(threads_used=self.num_threads)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._crawl_worker, max_depth) for _ in
                       range(self.num_threads)]
            try:
                concurrent.futures.wait(futures, timeout=None,
                                        return_when=concurrent.futures.ALL_COMPLETED)
            except KeyboardInterrupt:
                self.stop_event.set()
                print("Interrupted by user. Stopping crawling.")
            finally:
                stats.end_session(len(self.visited), len(self.links_seen))

    def _crawl_worker(self, max_depth):
        while not self.url_queue.empty() and self.max_pages > 0 and not self.stop_event.is_set():
            url, depth = self.url_queue.get()
            if depth > max_depth or url in self.visited:
                continue
            try:
                self.driver.get(url)
                time.sleep(2)
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                if soup:
                    self._process_page(url, soup)
                    links = self._extract_links(soup)
                    with self.lock:
                        self.visited.add(url)
                        self.max_pages -= 1
                        self.links_seen.update(links)
                    for link in links:
                        if not self.stop_event.is_set() and self.max_pages > 0:
                            self.url_queue.put((link, depth + 1))
            except Exception as e:
                print(f"Error crawling {url}: {e}")
                raise RuntimeError

    def _process_page(self, url, soup):
        text = soup.get_text()
        with self.lock:
            if self.max_pages > 0:
                self.graph.add_node(url, text=text)
                if isinstance(text, str):
                    text = self.text_processor.clean_content(content=text)
                    self.db.save_document_and_vector(url, text)
                else:
                    print(f"Unexpected type for text: {type(text)}")
                for link in self._extract_links(soup):
                    self.graph.add_edge(url, link)
                    print(f"Adding edge from {url} to {link}")

    def _extract_links(self, soup):
        all_links = set()
        # Regex patterns for filtering
        php_pattern = re.compile(r'\bphp\b', re.IGNORECASE)
        index_html_pattern = re.compile(r'\bindex\.html\b', re.IGNORECASE)

        # Extract links from static HTML
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith(('http://', 'https://')):
                if php_pattern.search(href) or index_html_pattern.search(href):
                    all_links.add(href)
                elif not href.startswith(('http://', 'https://')) and not href.startswith('#'):
                    # Handle relative URLs
                    full_url = self._resolve_relative_url(href)
                    if full_url:
                        all_links.add(full_url)

        # Check for JavaScript-generated links
        dynamic_links = self._get_dynamic_links()
        all_links.update(dynamic_links)
        return all_links

    def _resolve_relative_url(self, href):
        try:
            # Resolve relative URL to absolute URL
            resolved_url = self.driver.current_url.rstrip('/') + '/' + href.lstrip('/')
            return resolved_url
        except Exception as e:
            print(f"Error resolving URL: {e}")
            return None

    def _get_dynamic_links(self):
        dynamic_links = set()
        try:
            # Wait for the page to load and execute JavaScript
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )
            # Extract all links after JavaScript execution
            elements = self.driver.find_elements(By.TAG_NAME, 'a')
            for element in elements:
                href = element.get_attribute('href')
                if href and href.startswith(('http://', 'https://')):
                    dynamic_links.add(href)
        except Exception as e:
            print(f"Error fetching dynamic links: {e}")
        return dynamic_links

    def get_documents(self):
        return {url: self.graph.nodes[url]['text'] for url in self.graph.nodes}

    def get_pagerank(self):
        pagerank = nx.pagerank(self.graph)
        return pagerank

    def __del__(self):
        self.driver.quit()  # Clean up the WebDriver

def prune_all():
    db = VectorizedDatabase()
    stats = Statistics()
    db.prune_documents()
    db.prune_cache()
    db.prune_vectors()
    stats.prune_statistics()


def main(i=None, url=None):
    if i == 1:
        prune_all()
        return 1

    import sys
    from statistics import Statistics

    def signal_handler(sig, frame):
        print('Interrupt received, stopping crawling...')
        crawler.stop_event.set()
        stats.print_most_recent_statistics()
        stats.total_time_crawling()
        stats.current_expected_time()
        sys.exit(0)

    signal(SIGINT, signal_handler)
    stats = Statistics()
    crawler = SimpleWebCrawler(VectorizedDatabase())

    N, c = random.randint(1, 16), log2(10)
    diver_limit = floor(pow(10, 2*N))
    depth = floor(c*(2*N - 1))
    print(f"The limit for this deep-dive is: {diver_limit}, the depth for this dive is: {depth}")

    crawler.auto_crawl(limit=diver_limit, max_depth=depth, seed_url=url)
    stats.print_most_recent_statistics()
    stats.total_time_crawling()
    stats.current_expected_time()


if __name__ == '__main__':
    main(url='https://www.bbc.com')
