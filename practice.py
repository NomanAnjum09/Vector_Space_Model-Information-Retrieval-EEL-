from __future__ import print_function	# For Py2/3 compatibility
import eel
from nltk.stem import WordNetLemmatizer 
import math  
from nltk.stem.porter import *
from PyQt5 import QtCore, QtGui, QtWidgets
import nltk
nltk.download('wordnet')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer() 
Ltoken = []
Lposting = []
doc_freq = None
try:
    file = open('./sortedPosting/sort.txt')
    doc_freq = file.read().split('\n')
    file.close()
except:
    pass


######Vctor Generater#########O(n1+n2)
def generate( ref_words, doc_freq, tokens, tf, i):  #Genrates Vector
    print("Generating Vector {}".format(i))
    n1 = len(ref_words)               #Vector Length I equal to total no of terms
    n2 = len(tokens)                    #Terms length in current docs
    dot = 0
    n = 56
    j = 0
    k = 0

    file = open('./sortedPosting/vector_{}.txt'.format(i), 'w')

    while(k < n2 and j < n1 and tokens[k] != ''):   #Make vector untill terms in current doc ends 
        if(ref_words[j] != tokens[k]):
            file.write(str(0))
            file.write('\n')
            j += 1
        else:
            weight = float(tf[k])*math.log10(int(doc_freq[j])/n)
            print("{},{}".format(ref_words[j],weight),end=', ')
            dot += pow(weight, 2)                               #Calculation of tf^2 for getting Normalization
            j += 1
            k += 1
            file.write(str(weight))
            file.write('\n')
    while(j < n1):                                         #cat remaining term weights as zero
        if(ref_words[j] == ''):
            j += 1
            continue
        file.write('0')
        file.write('\n')
        j += 1
    while(k < n2):
        if(tokens[k] == ''):
            k += 1
            continue
        print(tokens[k])
        k += 1
    file.close()
    print("\nSquare sum of vector weights = {}".format(dot))
    return dot

#############Vector GEnerator  O(56)#####################################
def generate_vector():
    print("In funtion vector genrator")
    file = open('./sortedToken/sort.txt', 'r')
    data = file.read()
    ref_words = data.split('\n')
    file.close()
    file = open('./sortedPosting/sort.txt', 'r')
    file1 = open('./sortedPosting/normalDocs.txt', 'w')

    data = file.read()
    doc_freq = data.split('\n')
    file.close()
    print("Sending Term Frequencies For Generaring Vector")
    for i in range(56):                                       #Open all docs and send all terms and terms in current doc for vector generation
        ST = open('./sortedToken/sort_{}.txt'.format(i), 'r') #read terms
        SP = open('./sortedPosting/sort_{}.txt'.format(i), 'r')  #read correponsing term freq
        data = ST.read()
        tokens = data.split('\n')
        data = SP.read()
        tf = data.split('\n')

        dot = generate(ref_words, doc_freq, tokens, tf, i)
        file1.write(str(math.sqrt(dot)))
        file1.write('\n')
        ST.close()
        SP.close()
    file1.close()


######################mergesort  O(n)###################################


def merge( arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m

    L = [0] * (n1)
    R = [0] * (n2)

# Copy data to temp arrays L[] and R[]
    for i in range(0, n1):
        L[i] = arr[l + i]
    for j in range(0, n2):
        R[j] = arr[m + 1 + j]
# Merge the temp arrays back into arr[l..r]
    i = 0     # Initial index of first subarray
    j = 0     # Initial index of second subarray
    k = l     # Initial index of merged subarray

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def mergeSort( arr, l, r):
    if l < r:
        m = int((l+(r-1))/2)
        mergeSort(arr, l, m)
        mergeSort(arr, m+1, r)
        merge(arr, l, m, r)

def mergelists( x1, x2, ind, doc_freq):
    n1 = len(x1)
    n2 = len(x2)
    word = []
    freq_list = []
# Merge the temp arrays back into arr[l..r]
    i = 0     # Initial index of first subarray
    j = 0     # Initial index of second subarray
    k = 0     # Initial index of merged subarray

#word->dictionary, Plist->PostingList, posting-> <DocumentNumber>position of word
    while i < n1 and j < n2:
        if(x1[i] == ''):
            i += 1
            continue
        if(x2[j] == ''):
            j += 1
            continue

        if x1[i] < x2[j]:
            word.append(x1[i])  # Send Word To Dictionary
            freq_list.append(doc_freq[i])  # Append Previous Frequency
            i += 1
        elif x1[i] > x2[j]:  # Attach Document and Position since Second list's doc is smaller
            word.append(x2[j])
            freq_list.append(1)  # New Word Append 1 as frequency
            j += 1
        else:

            # Documewnts are same so join <docNo> and both postion Lists
            word.append(x1[i])
            freq_list.append(doc_freq[i]+1)  # Join Previous Frequency + 1
            i += 1
            j += 1

# Copy the remaining elements of list1, if there
# are any
    while i < n1:
        if(x1[i] == ''):
            i += 1
            continue
        word.append(x1[i])
        freq_list.append(doc_freq[i])
        i += 1

# Copy the remaining elements of list2, if there
# are any
    while j < n2:
        if(x2[j] == ''):
            j += 1
            continue
        word.append(x2[j])
        freq_list.append(1)
        j += 1
    return word, freq_list

def tokenizer():
    global err
    global Ltoken
    stopwords = []
    s = open('Stopword-List.txt', 'r')  # picking stopwords
    stopdoc = s.read()
    w = ''
    for i in range(len(stopdoc)):
        if(stopdoc[i] == '\n'):
            if(w != ''):
                stopwords.append(w)  # parsing stopwords
            w = ''
        elif stopdoc[i] != ' ':
            w += stopdoc[i]
    s.close()
#Tokenize stem fold case of words

    for i in range(56):
        print("Toeknizing Doc {}".format(i))
        f = open('./Trump Speechs/speech_{}.txt'.format(i))
        if f.mode == 'r':
            content = f.read()
            
            w = ''
            for j in range(len(content)):

                if((content[j] in [' ', '.', '\n',']','-','?']) and w != '' and w != "'" or (content[j-1]>='a' and content[j-1]<='z' and (content[j]>'z' or content[j]<'a'))):
                    # removing stopwords
                    if(w not in stopwords and w not in ['']):
                        #wor = stemmer.stem(w)
                        tk = lemmatizer.lemmatize(w)  # Lemmatization
                        Ltoken.append(tk)

                    w = ''

                elif content[j] not in ['', ' ', '[', ',', ':', '?', '(',')','—','"',';',"'",'!','-','.','\n']:
                    if(content[j] >= 'A' and content[j] <= 'Z'):  # Case folding
                        w = w+chr(ord(content[j])+32)
                    else:
                        w += content[j]

        # Sorting and adding frequency of Tokens In file
        print("Sorting Tokens Withing Document")
        r = len(Ltoken)-1
        mergeSort(Ltoken, 0, r)

        ST = open('./sortedToken/sort_{}.txt'.format(i), 'w')
        SP = open('./sortedPosting/sort_{}.txt'.format(i), 'w')
        counter = 1
        # Write token and tf to if no preceding word is same as current
        print("Removing Duplicate and Adding termfrequency")
        for l in range(0, len(Ltoken)-1):
            if Ltoken[l] != Ltoken[l+1]:
                ST.write(Ltoken[l])
                SP.write(str(counter/r))
                ST.write('\n')
                SP.write('\n')
                counter = 1
            else:
                counter += 1  # preceding word is same increase tf

        ST.write(Ltoken[len(Ltoken)-1])
        SP.write(str(counter))
        Ltoken.clear()
        ST.close()
        SP.close()

#Document as a BLOCK sorting done

def Processor():
    print("IN funtion Processor")
    ST = open('./sortedToken/sort_{}.txt'.format(0), 'r')
    txt = ST.read()
    x1 = txt.split('\n')
    doc_freq = [1]*len(x1)
    ST.close()
    for i in range(1, 56):
        print("Merging Doc {} with corpus".format(i))
        ST = open('./sortedToken/sort_{}.txt'.format(i), 'r')
        txt = ST.read()
        x2 = txt.split('\n')
        x1, doc_freq = mergelists(x1, x2, i, doc_freq)
        ST.close()

    file = open('./sortedToken/sort.txt', 'w')
    for i in range(len(x1)):
        file.write(x1[i])
        file.write('\n')
    file.close()
    file = open('./sortedPosting/sort.txt', 'w')
    for i in range(len(doc_freq)):
        file.write(str(doc_freq[i]))
        file.write('\n')
    file.close()

    generate_vector()

def merge_doc( arr, pos, l, m, r):
    n1 = m - l + 1
    n2 = r - m

    L = [0] * (n1)
    R = [0] * (n2)
    LP = [0] * (n1)
    RP = [0] * (n2)
# Copy data to temp arrays L[] and R[]
    for i in range(0, n1):
        L[i] = arr[l + i]
        LP[i] = pos[l + i]
    for j in range(0, n2):
        R[j] = arr[m + 1 + j]
        RP[j] = pos[m + 1 + j]

# Merge the temp arrays back into arr[l..r]
    i = 0     # Initial index of first subarray
    j = 0     # Initial index of second subarray
    k = l     # Initial index of merged subarray

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            pos[k] = LP[i]
            i += 1
        else:
            arr[k] = R[j]
            pos[k] = RP[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = L[i]
        pos[k] = LP[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        pos[k] = RP[j]
        j += 1
        k += 1

def sort_doc( arr, pos, l, r):
    if l < r:
        m = int((l+(r-1))/2)
        sort_doc(arr, pos, l, m)
        sort_doc(arr, pos, m+1, r)
        merge_doc(arr, pos, l, m, r)

def parse_Query( query):
    stopwords = []
    s = open('Stopword-List.txt', 'r')
    stopdoc = s.read()
    w = ''
    print("Removing StopWords")
    for i in range(len(stopdoc)):  # Get Stopwords to remove from query
        if(stopdoc[i] == '\n'):
            if(w != ''):
                stopwords.append(w)
            w = ''
        elif stopdoc[i] != ' ':
            w += stopdoc[i]
    s.close()
    parsed = []
    w = ''
    for i in range(len(query)):  # parse Query
        if(query[i] == ' '):
            if(w not in ['', ",", ' ']):
                parsed.append(w)
            w = ''
        elif query[i] not in [' ', "'", '"', '']:
            w += query[i]
    s.close()
    if w not in ['', ",", ' ']:
        parsed.append(w)
    for i in range(len(parsed)-1, -1, -1):
        if(parsed[i] in stopwords):
            print("{} removed".format(parsed[i]))
            parsed.pop(i)
        else:
            parsed[i] = parsed[i].lower()
            parsed[i] = lemmatizer.lemmatize(parsed[i])
    print("Unsorted Query =  {}".format(parsed))
    parsed.sort()
    print("Sorted Query = {}".format(parsed))
    return parsed

def binarySearch( arr, l, r, x):

    while l <= r:

        mid = int(l + (r - l)/2)

    # Check if x is present at mid
        if arr[mid] == x:
            return mid

    # If x is greater, ignore left half
        elif arr[mid] < x:
            l = mid + 1

    # If x is smaller, ignore right half
        else:
            r = mid - 1

    # If we reach here, then the element was not present
    return -1

def process_Query( parsed):
    file = open('./sortedToken/sort.txt', 'r')
    tokens = file.read().split('\n')                #read toekns from query
    count = 1
    new_parsed = []
    query_tf = []
    for i in range(len(parsed)-1):                  #if terns in query is redundant just add 1 to count
        if(parsed[i] == parsed[i+1]):
            count += 1
        else:
            new_parsed.append(parsed[i])            #append single term and tf 
            query_tf.append(count)
            count = 1
    new_parsed.append(parsed[len(parsed)-1])
    query_tf.append(count)
    result = []
    for i in range(len(new_parsed)):               #Find occurence of query terms in corpus
        result.append(binarySearch(
            tokens, 0, len(tokens)-1, parsed[i]))

    return result, query_tf, new_parsed


def fetch_docs( parsed, query_tf, results):
    #print(query_tf)
    #print(results)
    global doc_freq
    query_vector = []
    file = open('./sortedPosting/sort.txt')    # Open Doc frequency of all terms
    doc_freq = file.read().split('\n')
    file.close()
    weight = []
    lenght = len(parsed)
    print("Genrating Vector From Query")
    for i in range(len(results)):     #Making Vector from query term tf
        print("Document Frequency of {} = {}".format(
            parsed[i], doc_freq[results[i]]))
        query_vector.append(math.log10(
            int(doc_freq[results[i]])/56)*query_tf[i]/lenght)
        print("{}-->{}".format(parsed[i],math.log10(int(doc_freq[results[i]])/56)*query_tf[i]))
    print("------------")
    print("Query Vector")
    print(query_vector)
    print("------------")
    file1 = open('./sortedPosting/normalDocs.txt', 'r')
    normal_vectors = file1.read().split('\n')
    for i in range(56):                                                         #Vector addition of term in query with terms in each doc. Only weights 
        file = open('./sortedPosting/vector_{}.txt'.format(i), 'r')             # of intersecting terms are fetched and operation is performed
        vector = file.read().split('\n')
        ans = 0
        a = 0
        print("Vector For Doc {}".format(i))
        print("<",end='')
        for j in range(len(query_vector)):
            print("{},".format(vector[results[j]]),end=' ')
            ans = ans+query_vector[j]*float(vector[results[j]])
            a += pow(query_vector[j], 2)
        print(">")
        b = float(normal_vectors[i])
        try:
            rslt = round(ans/(math.sqrt(a)*b), 10)
            print("<A><B>/|A||B| = {}/ |{}||{}|".format(ans,math.sqrt(a),b))
        except:
            rslt = 0
        weight.append(rslt)

    return weight


import copy
def entertain(text,cutoff,choice):
    

    parsed = parse_Query(text)
    if(choice==1):
        print("Using Preprocessed Data")
        pass
    else:
        print("Running From Scratch")              #Run from scratch if flag is on
        tokenizer()
        Processor()
    
    print(parsed)
    eel.say_hello_js("Processing Query ...")      #Send current status to frontend
    results,query_tf,parsed = process_Query(parsed)

    for i in range(len(results)-1,-1,-1): #Remove words from query which are not in corpus
        if(results[i]==-1):
            parsed.pop(i)
            results.pop(i)
            query_tf.pop(i)

    result = fetch_docs(parsed,query_tf,results)
    docs = []
    print("Result Of Cosine Calculation")
    
    for i in range(56):
        docs.append(i)
    sort_doc(result,docs,0,len(result)-1)
    print(list(zip(docs,result)))

    ans =[]
    counter =0
    print("Cutoff",end='->')
    print(float(cutoff))
    for i in range(len(result)):
        real_query = copy.deepcopy(parsed)
        text = ""
        if(result[i]>=float(cutoff)):
            text+="(Doc#{}--{})\n".format(docs[i],result[i])
            counter+=1

            #Fetch Summary   ###########This Part Is Extra And is done via python builtin methods
            file = open('./Trump Speechs/speech_{}.txt'.format(docs[i]))
            data = file.read().split('.')
            counter1 = 0
            flag=0
            for line in data:
                for k in range(len(real_query)-1,-1,-1):
                    if real_query[k] in line.lower() or lemmatizer.lemmatize(real_query[k]) in line.lower():
                        text+=line
                        text+='\n'
                        text+="                              -----------------------------------"
                        text+='\n'
                        real_query.pop(k)
                        counter1+=1
                        if(counter1==2):
                            flag=1
                            break
                if(flag==1):
                    break
            text+="\n\n"
            #ans.append("(doc{}-{})\n".format(docs[i],result[i]))
            ans.append(text)
    
    ans.reverse()
    print(counter)
    res = "Lenght = {}\n\n".format(counter)
    for a in ans:
        print(a)
        res+=a
        res+="                        ************************************************************\n"
    return res













# Set web files folder
eel.init('web')

@eel.expose                         # Expose this function to Javascript
def say_hello_py(query,cutoff,choice):
    global doc_freq
    if doc_freq==None or choice==0:
        eel.say_hello_js("Building Indexes ...")
        choice = 0
    result = entertain(query,cutoff,choice)
    return result


   # Call a Javascript function
eel.start('index.html', size=(800, 800))


