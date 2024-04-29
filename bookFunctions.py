from lxml import etree

def normalize_from_list(list_of_dicts): 
    chunk_list = []
    idxes = [] 
    for idx, item in enumerate(list_of_dicts): 
        text_chunks = splitText(item['text'], 20)
        chunk_list.append(text_chunks)
        idxes.append(idx)
    return text_cleaner.text_normalizer2(chunk_list, idxes)
def splitText(textList, chunkLength):
    '''Takes list of words and desired length of passages, and returns 
    a list of word lists (with each wordlist the length of the chunkLength).'''
    count = 0 
    chunk = [] 
    chunks = [] 
    for word in textList: 
        if count < chunkLength: 
            chunk.append(word)
            count += 1 
        else: 
            count = 0
            chunks.append(' '.join(chunk)) ## modified to convert chunks from lists to strings
            chunk = [] 
    return chunks
def parseTexts(fileList): 
    for f in fileList:
        try: 
            context = etree.iterparse(f)
            text = [] 
            for action, elem in context:
                if elem.tag == 'wd': 
                    text.append(elem.text)
            return text
        except: 
            continue 

### Functions for examining classifier results            
def mostInformTerms (model, topn=20): 
    return sorted(list(zip(range(len(model.coef_[0])), model.coef_[0])), key= lambda x:x[1], reverse=True)[:topn]
def docTermMatrix (features, vectorizer): 
    return pd.DataFrame(features.transpose(), index=vectorizer.get_feature_names(), )
def buildMeanDF(df, features, categoryName, chunkIDx):
    group1IDs = [] 
    group2IDs = [] 
    for idx, txt in enumerate(features): 
        if chunkIDx[idx] == categoryName:
            group1IDs.append(idx)
        else: 
            group2IDs.append(idx)
    df1 = df[group1IDs]
    df2 = df[group2IDs]
    dfMean1 = pd.DataFrame(df1.mean(1), index=df1.index) 
    dfMean2 = pd.DataFrame(df2.mean(1), index=df2.index) 
    return dfMean1, dfMean2
def topGroupWord (df1, df2, informList, vocab): 
    top1 = [] 
    top2 = [] 
    for pair in informList: 
        if float(df1.iloc[pair[0]]) > float(df2.iloc[pair[0]]): 
            top1.append((vocab[pair[0]], pair[1]))
        else: 
            top2.append((vocab[pair[0]], pair[1]))
    return top1, top2
