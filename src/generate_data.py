import wikipediaapi

def get_categories(wiki_wiki, page_name):
    page_py = wiki_wiki.page(page_name)
    categories = wiki_wiki.categories(page = page_py, clshow="!hidden")
    res = []
    for title in sorted(categories.keys()):
        cleaned_category = str(categories[title])[9:].lower().split(" ")[:-4]
        if (len(cleaned_category) <= 2):
            res.append(" ".join(cleaned_category))
    return res
wiki_wiki = wikipediaapi.Wikipedia('en')
print(get_categories(wiki_wiki, "Dog"))
