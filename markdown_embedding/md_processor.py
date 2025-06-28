import re
from langchain_community.document_loaders import UnstructuredMarkdownLoader, UnstructuredHTMLLoader
from langchain_text_splitters.base import TextSplitter
from langchain_core.documents import Document
from typing import List, Optional, Callable, Any, Dict


# 处理错误时 抛出的报错
class MarkdownResolveError(Exception):
    def __init__(
        self,
        error_type: str = '',
        error_file: str = '',
    ):
        self.error_type = error_type
        self.error_file = error_file
    
    def __repr__(self, ):
        return f"MarkdownResolveError: {self.error_type} resolve failed when processing file {self.error_file}"


# 基本的解析结果类
class MarkdownLinesDoc:
    def __init__(
        self,
        metadata,
        content_lines,
    ):
        self.metadata = metadata
        real_lines = []
        temp_line = ""
        # 合并以 '\' 换行的内容
        for line in content_lines:
            if line.endswith("\\"):
                temp_line += line.strip(" \\") + " "
            elif temp_line != "":
                temp_line += line.strip()
                real_lines.append(temp_line)
                temp_line = ""
            else:
                real_lines.append(line.strip())
        self.content_lines = real_lines
    
    def __repr__(self, ):
        return f"MarkdownLinesDoc(metadata={self.metadata}, content_lines={self.content_lines})"


class BlockSplitter(TextSplitter):
    def __init__(
        self,
        max_chunk_length: int = 512,
        context_length: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.max_chunk_length = max_chunk_length
        self.context_length = context_length
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    # 合并 metadata，重复的 key，值相同不管，值不同重命名一个 key 赋值，如果是列表的话直接扩展
    def merge_metadata(self, metadata_a: Dict, metadata_b: Dict):
        for key, value in metadata_b.items():
            if key not in metadata_a:
                metadata_a[key] = value
            elif isinstance(metadata_a[key], list) and isinstance(value, list):
                metadata_a[key] = list(set(metadata_a[key] + value))
            elif metadata_a[key] == value:
                pass
            elif value not in [None, '', [], {}, ()]:
                metadata_a[key+'_m'] = value

        return metadata_a
    
    def merge_or_split_blocks(self, blocks: List[MarkdownLinesDoc], extra_metadata: Dict={}) -> List[Document]:
        chunks = []
        current_chunk = ""
        cur_metadata = {}
        index = 1
        
        for block in blocks:
            content = "\n".join(block.content_lines)
            # 如果block本身超过最大长度，需要分割
            if self._length_function(content) > self.max_chunk_length:
                if current_chunk:  # 先把当前积累的chunk保存
                    # 如果有 id，在后面加上序号
                    save_metadata = extra_metadata
                    if 'id' in save_metadata:
                        save_metadata = save_metadata | {'id': save_metadata['id'] + "_body_" + str(index)}
                        index += 1
                    chunks.append(Document(
                        page_content=current_chunk,
                        metadata=save_metadata | cur_metadata
                    ))
                    current_chunk = ""
                    cur_metadata = {}
                
                # 分割超长block
                split_parts = self._split_long_block(content)
                for part in split_parts:
                    # 如果有 id，在后面加上序号
                    save_metadata = extra_metadata
                    if 'id' in save_metadata:
                        save_metadata = save_metadata | {'id': save_metadata['id'] + "_body_" + str(index)}
                        index += 1
                    chunks.append(Document(
                        page_content=current_chunk,
                        metadata=save_metadata | block.metadata
                    ))
            else:
                # 检查添加当前block是否会超过长度限制
                if current_chunk and self._length_function(current_chunk) + self._length_function(content) > self.max_chunk_length:
                    # 如果有 id，在后面加上序号
                    save_metadata = extra_metadata
                    if 'id' in save_metadata:
                        save_metadata = save_metadata | {'id': save_metadata['id'] + "_body_" + str(index)}
                        index += 1
                    chunks.append(Document(
                        page_content=current_chunk,
                        metadata=save_metadata | cur_metadata
                    ))
                    current_chunk = content
                    cur_metadata = block.metadata
                else:
                    if current_chunk:
                        current_chunk += content
                        cur_metadata = self.merge_metadata(cur_metadata, block.metadata)
                    else:
                        current_chunk = content
                        cur_metadata = block.metadata
        
        if current_chunk:
            # 如果有 id，在后面加上序号
            save_metadata = extra_metadata
            if 'id' in save_metadata:
                save_metadata = save_metadata | {'id': save_metadata['id'] + "_body_" + str(index)}
            chunks.append(Document(
                page_content=current_chunk,
                metadata=save_metadata | cur_metadata
            ))
        
        # 添加上下文
        chunks_with_context = self._add_context(chunks)
        
        return chunks_with_context
    
    def _split_long_block(self, block: str) -> List[str]:
        parts = []
        remaining = block
        
        while self._length_function(remaining) > self.max_chunk_length:
            split_pos = -1
            # 按照优先级查找分割点
            for sep in self.separators:
                if not sep:  # 空字符串表示任意位置
                    pos = self.max_chunk_length
                else:
                    pos = remaining.rfind(sep, 0, self.max_chunk_length + 1)
                
                # 把分割点放到符号之后
                if pos > 0:
                    split_pos = pos + len(sep)
                    break
            
            if split_pos == -1:  # 没找到合适分割点，强制分割
                split_pos = self.max_chunk_length
            
            part = remaining[:split_pos]
            parts.append(part)
            remaining = remaining[split_pos:]
        
        if remaining:
            parts.append(remaining)
        
        return parts
    
    def _add_context(self, chunks: List[Document]) -> List[Document]:
        chunks_with_context = []
        n_chunks = len(chunks)
        
        for i in range(n_chunks):
            current = chunks[i].page_content
            
            # 添加上文
            start_context = ""
            if i > 0:
                prev_chunk = chunks[i-1].page_content
                start_context = prev_chunk[-self.context_length:] if self._length_function(prev_chunk) > self.context_length else prev_chunk
            
            # 添加下文
            end_context = ""
            if i < n_chunks - 1:
                next_chunk = chunks[i+1].page_content
                end_context = next_chunk[:self.context_length] if self._length_function(next_chunk) > self.context_length else next_chunk
            
            chunk_with_context = start_context + current + end_context
            chunks_with_context.append(Document(
                page_content=chunk_with_context,
                metadata=chunks[i].metadata
            ))
        
        return chunks_with_context

    def split_text(self, text: str) -> List[str]:
        # 如果输入是单个文本而不是blocks，先按最大长度分割
        return self._split_long_block(text)


"""
专门针对Arxiv从LaTex转html再转Markdown设计的分割器

关键点
pandoc's markdown文件使用::: xxx :::作为一个block
且外层比内层少
例如 :::: xxx ::: xxx ::: x ::::
此外也存在markdown原本的标题 #
这部分标题暂未处理

"""
class ArxivPandocMarkdownSplitter:
    def __init__(
        self, 
        block_splitter: BlockSplitter = None,
        max_chunk_length: int = 512,
        context_length: int = 200,
        **kwargs: Any
    ):
        self.re_patterns = {
            'intro': [
                r"##.*[1I].* {#\]",
                r"##.*[Ii][Nn][Tt][Rr][Oo][Dd][Uu][Cc][Tt][Ii][Oo][Nn].*{#",
                r"\[.*[Ii][Nn][Tt][Rr][Oo][Dd][Uu][Cc][Tt][Ii][Oo][Nn].*]{#",
            ],
            'title_key': r"{#.*\.ltx_title_document.*}",
            'abstract_key': [r"(?i)##\s*abstract", r"(?i)\[\s*abstract\s*", r"(?i)abstract\s*:"],
            'title_page_path_key': r"{#.*\.ltx_titlepage.*}",
            'author_path_key': 'ltx_authors',
            'dates_path_key': 'ltx_dates',
            'abstract_path_key': 'ltx_abstract',
            'classification_path_key': 'ltx_classification',
            'keywords_path_key': 'ltx_keywords',
            'acknowledgements_path_key': 'ltx_acknowledgements',
            'sacle_key': r"style=\"font-size:(\d*)%;\"",
            'bold_key': '.ltx_font_bold'
        }
        if not block_splitter:
            self.block_splitter = BlockSplitter(
                max_chunk_length=max_chunk_length,
                context_length=context_length,
                **kwargs
            )
        else:
            self.block_splitter = block_splitter
    
    @classmethod
    def from_huggingface_tokenizer(cls, tokenizer: Any, **kwargs: Any):
        block_splitter = BlockSplitter.from_huggingface_tokenizer(tokenizer, **kwargs)
        return cls(block_splitter, **kwargs)

    # 按行读文件
    def load_file(self, file_path):
        self.processing_file = file_path
        with open(file_path, "r") as f:
            lines = f.readlines()
            return lines
    

    # 只保留 page content 块，删除 logo
    def remove_logo_section(self, lines):
        block_s = None
        start_index = 0
        end_index = -1
        for i, line in enumerate(lines):
            if line.strip().endswith("ltx_page_content"):
                block_s = re.search(r'(:::+)', line).group(1)
                start_index = i + 1
            if block_s is not None and line.strip() == block_s:
                end_index = i
                break
        return lines[start_index: end_index]
    
    # 清除文中的 \hspace{0pt} \xa0 \u3000 \u2002 \u2003
    def clean_meaningless(self, line):
        line = re.sub(r"\\hspace\{0pt\}", "", line).strip()
        line = re.sub(r'[\xa0\u3000\u2002\u2003]', ' ', line)
        line = line.strip(' \n')
        return line

    # 递归切开所有块
    def split_recur(self, content_lines, block_path=[]):
        lines_split = []
        block_s = None
        start_index = 0
        cur_path = ""
        text_lines = []

        for i, line in enumerate(content_lines):
            # 寻找块开头
            if block_s is None:
                # 如果是块开头
                if line.startswith(":::"):
                    if text_lines != []:
                        temp_lines = []
                        for temp_line in text_lines:
                            temp_line = self.clean_meaningless(temp_line).strip()
                            if temp_line != '':
                                temp_lines.append(temp_line)
                        if text_lines != []:
                            new_block = MarkdownLinesDoc({'block_path': ['/'.join(block_path)]}, temp_lines)
                            if len(new_block.content_lines) != 0:
                                lines_split.append(new_block)
                        text_lines = []
                    block_s = re.search(r'(:::+)', line).group(1)
                    cur_path = re.sub(r'(:::+)', '', line).strip()
                    start_index = i + 1
                # 如果是普通文本
                elif line.strip() != '':
                    text_lines.append(line)
            # 寻找块结尾
            elif block_s is not None and line.strip() == block_s:
                temp = content_lines[start_index: i]
                # 如果有内容，在内部递归继续找块，没有块就处理纯文本
                if temp != []:
                    temp = self.split_recur(temp, block_path+[cur_path])
                    lines_split.extend(temp)
                    block_s = None
                else:
                    block_s = None
        
        # 处理块后文本        
        if text_lines != []:
            temp_lines = []
            for temp_line in text_lines:
                temp_line = self.clean_meaningless(temp_line).strip()
                if temp_line != '':
                    temp_lines.append(temp_line)
            if text_lines != []:
                new_block = MarkdownLinesDoc({'block_path': ['/'.join(block_path)]}, temp_lines)
                if len(new_block.content_lines) != 0:
                    lines_split.append(new_block)

        return lines_split

    # 去除所有的 latexml 格式 []{#id |.ltx}
    def clean_latex_format(self, text):
        pattern = r'\[(.*?)\]\{(?:\.|#)[^}]*\}'
        while True:
            new_text = re.sub(pattern, r'\1', text)
            if new_text == text:  # 如果没有变化，终止循环
                break
            text = new_text
        text = re.sub(r'\{#[^}]*\}', '', text)
        return text

    # 解析切分后的 blocks
    def resolve_blocks(self, blocks, file_path):
        if len(blocks) <= 0:
            raise MarkdownResolveError("Content Empty", file_path)
        
        raw_info = ''
        main_body_blocks = None

        title = ''
        author = ''
        dates = ''
        abstract = ''
        keywords = ''
        acknowledgements = ''

        abs_index = -1
        abs_path = ''

        avail_abs_index = -1
        title_page_end_index = -1
        Intro_index = -1
        meta_index = -1

        split_index = -1

        for i, block in enumerate(blocks):
            # 如果已经用 Intro 分割了就不再处理信息
            if Intro_index == -1:
                # 首先看路径中是否可以找到 title_page
                path = block.metadata['block_path'][0]
                if re.search(self.re_patterns['title_page_path_key'], path) is not None:
                    title_page_end_index = i + 1
                # 顺便找路径中的 ltx_authors ltx_dates ltx_abstract ltx_classification ltx_keywords ltx_acknowledgements
                if re.search(self.re_patterns['author_path_key'], path) is not None:
                    author += "\n".join(block.content_lines)
                    meta_index = i + 1
                elif re.search(self.re_patterns['dates_path_key'], path) is not None:
                    dates += "\n".join(block.content_lines)
                    meta_index = i + 1
                elif re.search(self.re_patterns['abstract_path_key'], path) is not None:
                    abstract += "\n".join(block.content_lines)
                    meta_index = i + 1
                elif re.search(self.re_patterns['classification_path_key'], path) is not None:
                    keywords += "\n".join(block.content_lines)
                    meta_index = i + 1
                elif re.search(self.re_patterns['keywords_path_key'], path) is not None:
                    keywords += "\n".join(block.content_lines)
                    meta_index = i + 1
                elif re.search(self.re_patterns['acknowledgements_path_key'], path) is not None:
                    acknowledgements += "\n".join(block.content_lines)
                    meta_index = i + 1

                # 其次看是否可以找到 Intro
                if title_page_end_index == -1:
                    for pattern in self.re_patterns['intro']:
                        if re.search(pattern, block.content_lines[0]):
                            Intro_index = i
                            break
                
                # 在内容中找 title、author、abstract 等
                for data in block.content_lines:
                    if title == '' and re.search(self.re_patterns['title_key'], data) is not None:
                        title = re.sub(r'#', '', self.clean_latex_format(data)).strip()
                    if abstract == '':
                        for pattern in self.re_patterns['abstract_key']:
                            if re.search(pattern, data):
                                abs_index = i
                                abs_path = block.metadata['block_path'][0]
                                abstract += self.clean_latex_format(data).strip()
                    elif abs_index != -1:
                        if block.metadata['block_path'][0].startswith(abs_path):
                            abstract += '\n' + self.clean_latex_format(data).strip()
                            abs_index = i
        
        # 如果从内容中找到了摘要，保存摘要最后的位置
        if abs_index != -1 and len(abstract) > 16:
            avail_abs_index = abs_index + 1
        
        # 所有 meta 信息的下一个 | 从 Intro 开始 | 从 title_page 块划分 ，这些都可以割开信息和正文
        split_index = max(avail_abs_index, title_page_end_index, Intro_index, meta_index)

        # 如果已经有摘要了就清洗一下，与后面保持对齐
        abstract = self.clean_latex_format(abstract).strip()

        # 如果能通过 摘要 之外的途径确定分割点，且 摘要 还没找到，
        # 那就从切分点之前按更细致的逻辑筛一遍
        if split_index != -1:
            # 筛摘要 从其他 meta 信息开始，直到分割点都算成摘要
            if abstract == '' and meta_index != -1:
                for i in range(meta_index, split_index):
                    for line in blocks[i].content_lines:
                        abstract += self.clean_latex_format(line).strip() + '\n'
            elif abs_index != -1 and len(abstract) > 0 and len(abstract) <= 16:
                for i in range(abs_index + 1, split_index):
                    for line in blocks[i].content_lines:
                        abstract += '\n' + self.clean_latex_format(line).strip()

        # 筛标题 找 最早的加粗字体 和 最早的最大字体 他们中最大的就是标题
        if title == '':
            max_scale = 0
            scale_title_indexes = []
            bold_title_indexes = []
            bold_last_index = -1
            scale_last_index = -1
            current_index = 0

            final_indexes = []
            for i, block in enumerate(blocks[:split_index]):
                for j, line in enumerate(block.content_lines):
                    res = re.search(self.re_patterns['sacle_key'], line)
                    if res is not None:
                        res = int(res.group(1))
                        if res > max_scale:
                            scale_title_indexes = [(i, j)]
                            scale_last_index = current_index
                            max_scale = res
                        if res == max_scale and scale_last_index == current_index - 1:
                            scale_title_indexes.append((i, j))
                            scale_last_index = current_index
                    
                    if (bold_last_index == -1 or bold_last_index == current_index - 1) and re.search(self.re_patterns['bold_key'], line):
                        bold_title_indexes.append((i, j))
                        bold_last_index = current_index

                    current_index += 1
            if bold_title_indexes == []:
                final_indexes = scale_title_indexes
            elif scale_title_indexes == []:
                final_indexes = bold_title_indexes
            else:
                if bold_title_indexes[0] == scale_title_indexes[0]:
                    final_indexes = sorted(list(set(bold_title_indexes) & set(scale_title_indexes)))
                elif bold_title_indexes[0] > scale_title_indexes[0]:
                    final_indexes = scale_title_indexes
                else:
                    final_indexes = bold_title_indexes

            if final_indexes != []:
                lines = []
                for indexes in final_indexes:
                    line = blocks[indexes[0]].content_lines[indexes[1]]
                    line = self.clean_latex_format(line).strip()
                    lines.append(line)
                    split_index = indexes[0]
                title = " ".join(lines).strip()
        
        if split_index != -1:
            raw_info = str(blocks[:split_index])
            main_body_blocks = blocks[split_index:]
        else:
            main_body_blocks = blocks
        
        file_id = re.sub(r"\.md", "", file_path.rsplit('/', 1)[-1])

        num_in_file = re.search(r"\d+", file_id).group(0)
        publish_time = ''
        if num_in_file is not None:
            publish_time = num_in_file[:2] + '-' + num_in_file[2:4]

        cur_metadata = {
            "id": file_id,
            "file_path": file_path,
            "file_id": file_id,
            "publish_time": publish_time,
            "title": title,
            "author": self.clean_latex_format(author).strip(),
            "dates": self.clean_latex_format(dates).strip(),
            "keywords": self.clean_latex_format(keywords).strip(),
            "acknowledgements": self.clean_latex_format(acknowledgements).strip(),
            "raw_info": raw_info,
        }

        return cur_metadata, abstract, main_body_blocks

    # 把当前层级的 block 割开
    def split_in_cur_level(self, lines):
        lines_split = []
        block_s = None
        start_index = 0
        for i, line in enumerate(lines):
            if block_s is None and line.startswith(":::"):
                block_s = re.search(r'(:::+)', line).group(1)
                start_index = i + 1
            if block_s is not None and line.strip() == block_s:
                temp = lines[start_index: i]
                if temp != []:
                    lines_split.append(temp)
                    block_s = None
        return lines_split

    # 统一的处理函数
    def process(self, file_path):
        raw_lines = self.load_file(file_path)

        content_lines = self.remove_logo_section(raw_lines)

        blocks = self.split_recur(content_lines)

        metadata, abstract, main_body_blocks = self.resolve_blocks(blocks, file_path)

        chunks = self.block_splitter.merge_or_split_blocks(main_body_blocks, metadata)

        abstract_chunks = []
        abstract_parts = self.block_splitter.split_text(abstract)
        for index, a_part in enumerate(abstract_parts):
            abstract_chunks.append(Document(
                page_content=a_part,
                metadata=metadata | {
                    'id': metadata['id'] + "_abs_" + str(index + 1),
                    'block_path': ['ltx_abstract']
                }
            ))
        
        return abstract_chunks, chunks


# 老版处理函数，暂时废弃
def process_md(md_file, max_length, test=False):
    md_loader = UnstructuredMarkdownLoader(md_file, mode="elements")
    md_elements = md_loader.load()
    
    if test:
        import pdb; pdb.set_trace()
    
    # 清洗无意义数据 Latexml Logo
    if md_elements[-3].page_content.startswith("::: ltx_page_logo") and md_elements[-1].page_content.startswith("{.ltx_LaTeXML_logo}"):
        md_elements = md_elements[:-3]

    # 切割正文和前置部分
    split = 0
    for i in range(len(md_elements)):
        if md_elements[i].page_content.endswith(".ltx_section}"):
            split = i
    try:
        assert split != 0
    except:
        return 0, "No Main Body Section"
    
    content_chunks = md_elements[split:]

    info_chunks = md_elements[:split]

    index = 0
    title = ""
    abstract = ""
    author_info = ""
    keywords = ""
    while index < len(info_chunks):
        # 确认 Title
        if info_chunks[index].page_content.endswith(".ltx_title_document}"):
            title = re.sub(r'\{.*?\}', '', info_chunks[index].page_content, flags=re.DOTALL)
        elif info_chunks[index].page_content.endswith("""style="font-size:144%;"}"""):
            res = re.search(r'\[(.*?)\]', info_chunks[index].page_content, flags=re.DOTALL)
            if res is not None:
                title += re.search(r'\[(.*?)\]', info_chunks[index].page_content, flags=re.DOTALL).group(1) + " "
        else:
            # 确认 Abstract
            if info_chunks[index].page_content.startswith("Abstract") or info_chunks[index].page_content.startswith("[ABSTRACT"):
                while not info_chunks[index].page_content.endswith(":::"):
                    abstract += info_chunks[index].page_content + "\n"
                    index += 1
                abstract += info_chunks[index].page_content
            else:
                # 确认 Authors
                if info_chunks[index].page_content.startswith("::: ltx_authors") or info_chunks[index].page_content.endswith("""style="font-size:120%;"}"""):
                    while not info_chunks[index].page_content.endswith(":::"):
                        author_info += info_chunks[index].page_content + "\n"
                        index += 1
                    author_info += info_chunks[index].page_content
                else:
                    # 可能有 Keywords
                    if info_chunks[index].page_content.startswith("::: ltx_keywords"):
                        while not info_chunks[index].page_content.endswith(":::"):
                            keywords += info_chunks[index].page_content + "\n"
                            index += 1
                        keywords += info_chunks[index].page_content
        index += 1
    title = title.strip(" \\")
    if title == "":
        if test:
            import pdb; pdb.set_trace()
        return 0, "Title Resolve Failed"
    elif abstract == "":
        if test:
            import pdb; pdb.set_trace()
        return 0, "Abstract Resolve Failed"    
    elif author_info == "":
        if test:
            import pdb; pdb.set_trace()
        return 0, "Author Info Resolve Failed"
    
    if test:
        import pdb; pdb.set_trace()

    # 获取文章发布时间
    if "-" in md_elements[0].metadata['filename']:
        time = md_elements[0].metadata['filename'].split('-')[1][2:6]
    else:
        time = md_elements[0].metadata['filename'].split('.', 1)[0]
    time = time[:2] + "-" + time[2:]
    

    merged_chunks, metadatas, ids = merge_markdown_chunks(content_chunks, {'title': title, 'author_info': author_info, 'publish_time': time, 'keywords': keywords}, max_length)

    # 提取摘要并合并
    abstract_metadata = md_elements[3].metadata
    abstract_id = abstract_metadata['filename'].rsplit('.', 1)[0] + '.0'
    abstract_metadata.update({
        "id": abstract_id, 
        'category': 'Abstract', 
        'title': title, 
        'author_info': author_info, 
        'publish_time': time, 
        'keywords': keywords
    })
    abstract = md_elements[3].page_content + md_elements[4].page_content + md_elements[5].page_content

    merged_chunks.insert(0, abstract)
    metadatas.insert(0, abstract_metadata)
    ids.insert(0, abstract_id)

    return 1, merged_chunks, metadatas, ids


# 老版的合并code，目前废弃
def merge_markdown_chunks(chunks: list[str], general_metadata, max_length: int) -> list[str]:
    merged_chunks = []
    metadatas = []
    ids = []

    current_chunk = ""
    current_metadata = {}
    _id = chunks[0].metadata['filename'].rsplit('.', 1)[0]
    
    i = 1
    for chunk in chunks:
        # 如果当前块 + 新块不超过限制，则合并
        current_metadata = chunk.metadata
        if len(current_chunk) + len(chunk.page_content) <= max_length:
            current_chunk += "\n\n" + chunk.page_content if current_chunk else chunk.page_content
        else:
            if current_chunk:
                ids.append(_id + f'.{i}')
                i += 1
                merged_chunks.append(current_chunk)
                current_metadata.update(general_metadata)
                metadatas.append(current_metadata)
            current_chunk = chunk.page_content
    
    if current_chunk:  # 添加最后一个块
        ids.append(_id + f'.{i}')
        merged_chunks.append(current_chunk)
        current_metadata.update(general_metadata)
        metadatas.append(current_metadata)
    
    return merged_chunks, metadatas, ids


if __name__ == "__main__":
    # process_md("/RLS002/Public/arxiv/process_script/arxiv2markdown/output/astro-ph9912340/astro-ph9912340.md", max_length=2048, test=True)
    
    # file1 = "/RLS002/Public/arxiv/process_script/markdown2embeddings/example_markdown/__ok/introsplit/good/2505.11851.md"
    # file2 = "/RLS002/Public/arxiv/process_script/arxiv2markdown/output/astro-ph9912340/astro-ph9912340.md"
    # file3 = "/RLS002/Public/arxiv/process_script/arxiv2markdown/output/hep-ph9912397/hep-ph9912397.md"
    
    from transformers import AutoTokenizer

    model_name="/home/reallm/yuyang/embedding_models/Qwen3-Embedding-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splitter = ArxivPandocMarkdownSplitter.from_huggingface_tokenizer(
        tokenizer,
        max_chunk_length = 512,
        context_length = 200,
        separators = [" ", ""],
    )
    splitter.process("arxiv/process_script/markdown2embeddings/example_markdown/__ok/metaresolve/title/math9805118.md")

    # # 批量测试
    # path = "arxiv/process_script/markdown2embeddings/example_markdown/__ok/metaresolve"
    # import os
    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         if file.endswith(".md"):
    #             splitter.process(os.path.join(root, file), 1024)

    # text = r"[ [Sajin Vincent A.W. ]{.ltx_personname}[ [Indian Institute of Technology Indore, Khandwa Road, Simrol, Indore 453552, INDIA ]{.ltx_contact .ltx_role_address} [<sajinvincent2@gmail.com> ]{.ltx_contact .ltx_role_email}]{.ltx_author_notes}]{.ltx_creator .ltx_role_author} [,\xa0]{.ltx_author_before}[ [Aniruddha Deshmukh ]{.ltx_personname}[ [Indian Institute of Technology Indore, Khandwa Road, Simrol, Indore 453552, INDIA ]{.ltx_contact .ltx_role_address} [[aniruddha480@gmail.com](mailto:aniruddha480@gmail.com%20) ]{.ltx_contact .ltx_role_email}]{.ltx_author_notes}]{.ltx_creator .ltx_role_author} [\xa0and\xa0]{.ltx_author_before}[ [Vijay Kumar Sohani ]{.ltx_personname}[ [Indian Institute of Technology Indore, Khandwa Road, Simrol, Indore 453552, INDIA ]{.ltx_contact .ltx_role_address} [<vsohani@iiti.ac.in> ]{.ltx_contact .ltx_role_email} [ ]{.ltx_contact .ltx_role_dedicatory}]{.ltx_author_notes}]{.ltx_creator .ltx_role_author}"
    # print(remove_ltx_tags(text))