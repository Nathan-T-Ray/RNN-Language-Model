import re
def preprocess(text: str) -> str:
    """
    Preprocess the text for use by tokenizer. It should 
    separate punctuation like (!"#$%&'()*+,-./\\:;=?@[]^_`{|}~) 
    from words, lowercase the input, and remove any newline 
    characters, trailing spaces, or extra spaces. 
    Take care that the punctuation does not include < or >, 
    so that the special tokens (e.g., <unk>)
    are not modified and with / in </s>.

    Args:
        text (str): Text to be input to the tokenizer.

    Returns:
        str: Preprocessed text

    For example, 
        >>> from tokenizer import Tokenizer
        >>> tokenizer = Tokenizer()
        >>> tokenizer.preprocess(" the man, who is tall, is happy!\n")
        >>> "the man , who is tall , is happy !"
        >>> tokenizer.preprocess(" The <unk> cat loves to buy food; I love him? \n\n")
        >>> "the <unk> cat loves to buy food ; i love him ?"

    Hints: 
        For handling punctuation, you may consider using 
        regular expressions using pythons re package or
        the string functions translate and maketrans. 
        Consider, for example, wanting to replace B with ABC: 
        >>> text = 'CD B EF'
        >>> re.sub('B', r'ABC', text)
        >>> 'CD ABC EF'
        ...
        >>> text = 'CD B EF'
        >>> table = {'B': "ABC"}
        >>> text.translate(str.maketrans(table))
        >>> 'CD ABC EF'
    """
    text = text.strip().lower()


    special = '(' + '|'.join(['unk>', '/s>', 'pad>', 's>'])+')'

    text = re.sub(r"([^a-zA-Z0-9\s><])(?!"+special+')', r" \1 ", text)
    text = re.sub(r'\s{2,}', ' ', text).strip()

    return text


def tokenize(text: str) -> list[str]:
    """
    Takes a string a returns a list of tokens. Tokens are defined
    as space delineated characters. 

    Args: 
       text (str): text to be input to tokenizer.

    Returns:
        list[str]: A list of strings (words).

    For example, 
        >>> from tokenizer import Tokenizer
        >>> tokenizer = Tokenizer()
        >>> tokenizer.tokenize("the man , who is tall , is happy !")
        >>> ["the", "man", ",", "who", "is", "tall", ",", "is", "happy", "!"]
    """

    return text.split(' ')


def word_tokenize(text: str) -> list[str]:
    """
    Takes a string and returns a list of tokens. 

    Args: 
        text (str): input string

    Returns:
        list[str]: A list of tokens (words).

    For example, 
        >>> from tokenizer import Tokenizer
        >>> tokenizer = Tokenizer()
        >>> tokenizer.word_tokenize("the man, who is tall, is happy!")
        >>> ["the", "man", ",", "who", "is", "tall", ",", "is", "happy", "!"]
    """
    return tokenize(preprocess(text))

def get_vocab(files:list[str], freq:int = 20) -> list:
    """ Get vocab of most freq words from files (separating punctuation)

    Args:
        files (list): list of file names
        freq (int): frequency threshold

    Returns:
        vocab (list): most frequent words
    """
    freqs = {}
    for file in files:
        with open(file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                words = word_tokenize(line)
                for word in words:
                    freqs[word] = freqs.get(word, 0)+1
    vocab = []
    for word in freqs:
        if word == '':
            continue
        if freqs[word] > freq:
            vocab.append(word)
    for word in ["<unk>", "<s>", "<pad>"]:
        if word not in vocab:
            vocab.append(word)
    return vocab

if __name__ == "__main__":
    files = [f"../{split}/{split}.txt" for split in ["test", "train", "eval"]]
    vocab = get_vocab(files)
    with open('../vocab.txt', 'w') as f:
        for word in vocab:
            f.write(f"{word}\n")
