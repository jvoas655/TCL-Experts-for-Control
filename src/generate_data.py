import wikipediaapi
from selenium import webdriver
import time
import pickle

class Wikipedia_Scraper:

    def __init__(self, use_schrome_driver = True):
        self.xpath_sub_categories = "//*[(@id = \"mw-subcategories\")]//a"
        # self.xpath_pages = "//*[(@id = \"mw-pages\")]//a"
        # self.xpath_pages = "//*[contains(concat( \" \", @class, \" \" ), concat( \" \", \"mw-category-group\", \" \" ))]//a"
        self.xpath_pages = "//*[(@id = \"mw-pages\")]//*[contains(concat( \" \", @class, \" \" ), concat( \" \", \"mw-category-group\", \" \" ))]//a"
        self.wiki_wiki = wikipediaapi.Wikipedia('en')
        # TODO: replace filepath with relative file path
        if use_schrome_driver:
            self.driver = webdriver.Chrome(executable_path='/Users/alex3/Documents/UT/NLG/LiFT-Co-Expert-Text-Generation/chromedriver')

    def run_scraper(self, min_pages = 1000, max_depth = 3, output_file = "data/category_text_pairs", page_url_path = None):
        self.output_file = output_file
        if page_url_path != None:
            with open(page_url_path, 'rb') as f:
                page_urls = pickle.load(f)
        else:
            page_urls = self.get_pages( min_pages = min_pages, max_depth = max_depth)
            self.save_data(page_urls, output_file = "data/page_urls", format="pickle")
        category_to_text_pairs = self.make_category_text_pairs(page_urls)
        self.save_data(category_to_text_pairs, output_file = self.output_file, format="pickle")
        print("Done Scraping")

    def save_data(self, category_to_text_pairs, output_file, format="pickle"):
        if format == "pickle":
            with open(output_file, 'wb') as fp:
                pickle.dump(category_to_text_pairs, fp)
        else:
            raise NotImplementedError()

    def get_categories_from_category_page(self):
        time.sleep(.5)
        elements_categories = self.driver.find_elements("xpath", self.xpath_sub_categories)
        category_urls = []
        for elem in elements_categories:
            category_url = elem.get_attribute('href')
            category_urls.append(category_url)
        return category_urls

    def get_pages_from_category_page(self):
        time.sleep(.33) # limit of 200 calls per second
        elements_pages = self.driver.find_elements("xpath", self.xpath_pages)
        page_urls = []
        for elem in elements_pages:
            page_url = elem.get_attribute('href')
            page_urls.append(page_url)
        return page_urls

    def get_pages(self, min_pages = 10, max_depth = 3):
        self.driver.get("https://en.wikipedia.org/wiki/Category:Main_topic_classifications")
        all_page_urls = []
        curr_category_urls = self.get_categories_from_category_page()
        # creates mapping from int (representing depth) to empty list
        depth_to_category_urls = dict(zip(range(1, max_depth+1), [[] for _ in range(max_depth)] )) 
        depth_to_category_urls[1] = curr_category_urls[1::] # ignore first category

        # breadth first search for pages
        for depth in range(1, max_depth+1):
            category_urls_at_depth = depth_to_category_urls[depth]
            for category_url in category_urls_at_depth:
                if len(all_page_urls) > min_pages:
                    break
                time.sleep(.5)
                self.driver.get(category_url)
                categories_at_depth_plus_one = self.get_categories_from_category_page()
                depth_to_category_urls[depth+1].extend(categories_at_depth_plus_one)
                curr_page_urls = self.get_pages_from_category_page()
                all_page_urls.extend(curr_page_urls)
        return all_page_urls

    @staticmethod 
    def get_text_from_page(page):
        return page.text

    @staticmethod 
    def get_summary_from_page(page):
        return page.summary

    def make_category_text_pairs(self, page_urls):
        category_to_texts = {}
        for page_num, page_url in enumerate(page_urls):
            attempt_cnt = 0
            max_attempts = 5
            while attempt_cnt < max_attempts:
                attempt_cnt+=1
                page_category = page_url.split("/")[-1]
                time.sleep(.5)
                try:
                    categories = self.get_categories_from_page(page_category)
                    page = self.wiki_wiki.page(page_category)
                    text = Wikipedia_Scraper.get_text_from_page(page)
                    continue
                except Exception as e:
                    print(e)
            if attempt_cnt==max_attempts:
                print("Skipping {}".format(page_category))
                continue
            for category in categories:
                if category not in category_to_texts:
                    category_to_texts[category] = list() 
                category_to_texts[category].append(text)
            if page_num % 1000:
                self.save_data(category_to_texts, output_file = self.output_file, format="pickle")
        return category_to_texts

    def get_categories_from_page(self, page_name):
        page_py = self.wiki_wiki.page(page_name)
        categories = self.wiki_wiki.categories(page = page_py, clshow="!hidden")
        res = []
        for title in sorted(categories.keys()):
            cleaned_category = str(categories[title])[9:].lower().split(" ")[:-4]
            if (len(cleaned_category) <= 2):
                res.append(" ".join(cleaned_category))
        return res

# wiki_scraper = Wikipedia_Scraper(use_schrome_driver = True)
# wiki_scraper.run_scraper(min_pages = 60000, max_depth = 7, output_file = "data/category_text_pairs_large")

wiki_scraper = Wikipedia_Scraper(use_schrome_driver = False)
wiki_scraper.run_scraper(min_pages = 60000, max_depth = 7, output_file = "data/category_text_pairs_large", page_url_path="data/page_urls")
