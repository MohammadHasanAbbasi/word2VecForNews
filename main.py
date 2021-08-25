import tensorflow as tf
import numpy as np
import collections, math, os, random
import nltk
nltk.download('punkt')
from nltk.corpus import reuters,stopwords
import keras.preprocessing.text



train_fields = [f for f in reuters.fileids() if("train") in f]
test_fields = [f for f in reuters.fileids() if("test") in f]



train_raw = reuters.sents(train_fields)
train_raw_sents = [' '.join(item) for item in train_raw]


for i in range(len(train_raw_sents)):
    train_raw_sents[i]=train_raw_sents[i].lower()







train_data = list()
for i in range(len(train_raw_sents)):
    train_data.append(keras.preprocessing.text.text_to_word_sequence(train_raw_sents[i], filters='!"#$%&()1234567890*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '))
import re
def normalize_text(text):
    text=text.lower()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(pic\.twitter\.com/[^\s]+))','', text)
    text = re.sub('@[^\s]+','', text)
    text = re.sub('#([^\s]+)', '', text)
    text = re.sub('[:;>?<=*+()&,\-#!$%\{˜|\}\[^_\\@\]1234567890’‘]',' ', text)
    text = re.sub('[\d]','', text)
    text = text.replace(".", '')
    text = text.replace("'", '')
    text = text.replace("`", '')
    text = text.replace("'s", '')
    text = text.replace("/", ' ')
    text = text.replace("\"", ' ')
    text = text.replace("\\", '')
    
    text=re.sub( '\s+', ' ', text).strip()
    
    return text


train_normalized_sents=[]
for sentence in train_raw_sents:
    norm_sent=normalize_text(sentence)
    train_normalized_sents.append(norm_sent)
 
   

for i in range(len(train_normalized_sents)):
    train_normalized_sents[i]=train_normalized_sents[i].split()


voc_size=24718-1
words=list()
for sents in train_normalized_sents:
    words.extend(sents)
count= collections.Counter(words).most_common(voc_size) 
co=[]
for i in range(len(count)):
    co.append(count[i][0])

stop = stopwords.words('english')
words=list(filter(lambda x: x in co, words))
words=list(filter(lambda x: x not in stop, words))

count= collections.Counter(words).most_common() 
co=[]
for i in range(len(count)):
    co.append(count[i][0])

for i in range(len(train_normalized_sents)):
    train_normalized_sents[i]=list(filter(lambda x: x in co, train_normalized_sents[i]))
    train_normalized_sents[i]=list(filter(lambda x: x not in stop, train_normalized_sents[i]))



# Build dictionaries
unique_words =  [i[0] for i in count]
dic = {w: i for i, w in enumerate(unique_words)} #dic, word -> id cats:0 dogs:1 ......
voc_size = len(dic)





data = [dic[word] for word in words] #count rank for every word in words






#Window Size =2

skip_gram_pairs=list()


skip_gram_pairs_words = list()
for i in range(len(train_normalized_sents)):    
    for item in list(zip(train_normalized_sents[i][0:-1],train_normalized_sents[i][1:])):
        skip_gram_pairs_words.append(item )
    for item in list(zip(train_normalized_sents[i][0:-2],train_normalized_sents[i][2:])):
        skip_gram_pairs_words.append(item )

    k = list(ele for ele in reversed(train_normalized_sents[i]))
    for item in list(zip(k[0:-1],k[1:])):
        skip_gram_pairs_words.append(item )
    for item in list(zip(k[0:-2],k[2:])):
        skip_gram_pairs_words.append(item )
        
for i in range(len(skip_gram_pairs_words)):
    skip_gram_pairs.append(list([dic[skip_gram_pairs_words[i][0]],dic[skip_gram_pairs_words[i][1]]]))        
        

X_train=[]
Y_train=[]
    
for i in range(len(skip_gram_pairs)):
    Y_train.append(skip_gram_pairs[i][1])

for i in range(len(skip_gram_pairs)):
    X_train.append(skip_gram_pairs[i][0])  
X_train=np.array(X_train)
list_of_batch_size=[64,128]
list_of_embedding_size=[128,256]
list_of_neg_samples=[32,64]
filelist=["file1.txt","file2.txt","file3.txt","file4.txt","file5.txt","file6.txt","file7.txt","file8.txt"]
for batch_size in list_of_batch_size:
    for embedding_size in list_of_embedding_size:
        for num_sampled in list_of_neg_samples:

    
            X= tf.placeholder(tf.int32,shape=[None,]) #inputs
            Y= tf.placeholder(tf.int32,shape=[None,1]) #labels
            
            
                  #
            embeddings = tf.Variable(tf.random_normal([voc_size,embedding_size],-1.0,1.0))
            embed = tf.nn.embedding_lookup(embeddings, X) # lookup table
                
           
            nce_weights = tf.Variable(tf.random_normal([voc_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([voc_size]))
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,labels=Y,inputs=embed,num_sampled=num_sampled,num_classes=voc_size))

            
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
            n_iters = int(np.ceil(len(X_train)/batch_size))
            print(n_iters)
            Y_train = np.array(Y_train)
            
            epochs=101
            for cnt in range(epochs):
                index= 0
                for i in range(n_iters):
                    sess.run(optimizer, feed_dict={X: X_train[index:index+batch_size], Y: np.expand_dims(Y_train[index:index+batch_size],axis=1)})
                    index = index + batch_size
                if cnt % 10 == 0:
                    print('loss is : ', sess.run(loss, feed_dict={X: X_train[0:batch_size], Y: np.expand_dims(Y_train[0:batch_size],axis=1)}))
            
                    print('epoch %d done'%cnt)     
            
            
            trained_embeddings = np.array(sess.run(embeddings))
            
            
            
            
            
            
            
            
            
            
            
       
#            
            
            def euclidean_dist(vec1, vec2):
                return np.sqrt(np.sum((vec1-vec2)**2))
            
            def find_closest(word_index, vectors):
                min_dist = 10000 # to act like positive infinity
                min_index = -1
                query_vector = vectors[word_index]
                for index, vector in enumerate(vectors):
                    if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
                        min_dist = euclidean_dist(vector, query_vector)
                        min_index = index
                return min_index
            inv_dic = {v:k for k, v in dic.items()}
            def similar(text):
                i=dic[text];
                j=find_closest(i, trained_embeddings)
                print(inv_dic[j])
                
            
            
            
            em_val=trained_embeddings.tolist()
            for i,item in enumerate(em_val):
                item.insert(0,unique_words[i])
                
            FF=[" ".join(map(str,item)) for item in em_val] 
               
            file=open(batch_size.__str__()+"_"+num_sampled.__str__()+"_"+embedding_size.__str__()+".txt","w")

            
            for item in FF:
                file.write("%s\n" % item)
            file.close()
