import argparse
import os
from google_crawler import GoogleImageCrawler

parser = argparse.ArgumentParser()
parser.add_argument('keyword', help='Keyword to search')
args = parser.parse_args()

google_crawler = GoogleImageCrawler(
    parser_threads=2, 
    downloader_threads=4,
    storage={ 'root_dir': 'images/temp' }
)

print('keyword', args.keyword)
google_crawler.crawl(
    keyword=args.keyword, 
    offset=0, 
    max_num=8,
    min_size=(500, 500)
)

image_name = '_'.join(args.keyword.split(' '))
for index, filename in enumerate(os.listdir('./images/temp')):
    name, ext = os.path.splitext(filename)
    # print(name, ext)
    old_file = 'images/temp/' + filename
    new_file = 'images/raw/' + image_name + '_0_' + str(index + 1) + ext
    # print(old_file)
    # print(new_file)
    os.rename(old_file, new_file)
