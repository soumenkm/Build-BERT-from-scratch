import re, torch, requests, tqdm, pathlib, zipfile, tarfile, shutil

class GutenbergProcessor:
    
    def __init__(self, num_books: int, project_dir: pathlib.Path):
        
        self.num_books = num_books
        self.article_ids = list(range(73829, 0, -1))[:self.num_books]
        
        self.url = "https://gutenberg.org/cache/epub/{article_id}/pg{article_id}.txt"
        
        self.project_dir = project_dir
        pathlib.Path.mkdir(project_dir, parents=True, exist_ok=True) 
        
    def _download_single_book(self, article_id) -> None:
        
        src_url = self.url.format(article_id=article_id)
        destination_file_dir = pathlib.Path(self.project_dir, "raw")
        pathlib.Path.mkdir(destination_file_dir, parents=False, exist_ok=True) 
        destination_file_path = pathlib.Path(destination_file_dir, f"pg{article_id}.txt")
        
        response = requests.request(method="get", url=src_url, stream=True)

        status = response.status_code
        if status != 200:
            raise ValueError(f"Download unsuccessful. Status: {status}")
        
        headers = response.headers
        file_size = int(headers["Content-Length"]) if "Content-Length" in headers.keys() else 0
        chunk_size = 4096
        
        if pathlib.Path.exists(destination_file_path):
            if destination_file_path.stat().st_size == file_size:
                print(f"File already downloaded and matches the expected size ({file_size / (1024 ** 2):.2f} MB). Skipping download.")
                return None
            else:
                pathlib.Path.unlink(destination_file_path)
                print(f"File already downloaded but does not match the expected size ({file_size / (1024 ** 2):.2f} MB). Starting fresh download.")
        
        with open(file=destination_file_path, mode="w", encoding=response.encoding) as f: 
            with tqdm.tqdm(response.iter_content(chunk_size=chunk_size, decode_unicode=True),
                        desc="Downloading...",
                        total=file_size//chunk_size if file_size > 0 else None,
                        unit=f" chunk (1 chunk = {chunk_size//1024} KB)",
                        colour="green") as pbar:
                for i, chunk in enumerate(pbar):
                    if chunk:
                        f.write(chunk) # to ensire that None chunks are filtered out
                    pbar.set_postfix(downloaded=f"{chunk_size * i/1024**2:.2f} MB")
        
        print(f"Download is successfully completed! Total disk space used: {file_size / (1024 ** 2):.2f} MB")
        return None
    
    def download_corpus(self) -> None:
        
        for article_id in tqdm.tqdm(self.article_ids):
            self._download_single_book(article_id=article_id)
            
    def _remove_header(self, text: str) -> str:
        
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        start = text.find(start_marker) + len(start_marker)
        end = text.find(end_marker)
        
        text = text[start:end]
        text = re.sub(r'\n\s*\n', '\n', text).strip()
        
        return text

    def _clean_single_book(self, article_id: int) -> None:
        
        src_url = self.url.format(article_id=article_id)
        src_file_path = pathlib.Path(self.project_dir, f"raw/pg{article_id}.txt")
        
        dst_file_dir = pathlib.Path(self.project_dir, "clean")
        pathlib.Path.mkdir(dst_file_dir, parents=False, exist_ok=True)
        
        dst_file_path = pathlib.Path(dst_file_dir, src_file_path.name)
        with open(src_file_path, "r") as f:
            text = f.read()
        
        text = self._remove_header(text=text)
        with open(dst_file_path, "w") as f:
            f.write(text)
            
    def clean_corpus(self) -> None:
        
        for article_id in tqdm.tqdm(self.article_ids):
            self._clean_single_book(article_id=article_id)
      
if __name__ == "__main__":
    
    gb = GutenbergProcessor(num_books=50, project_dir=pathlib.Path(pathlib.Path.cwd(), "data/pre_training/gutenberg"))
    gb.download_corpus()
    gb.clean_corpus()