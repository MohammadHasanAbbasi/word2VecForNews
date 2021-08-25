import tensorflow as tf
import numpy as np
import nltk
import keras.preprocessing.text
import collections, math, os, random



file=open("questions-words.txt","r")


lines=list()
for i,line in enumerate(file):
    lines.append(line.lower())
lines=lines[1:]   
train_lines=list()
for line in lines:
    if line.split()[0] !=':':
        train_lines.append(line.split())

words=list()
for sents in train_lines:
    words.extend(sents)


count= collections.Counter(words).most_common() 


unique_words =  [i[0] for i in count]
dic = {w: i for i, w in enumerate(unique_words)} #dic, word -> id cats:0 dogs:1 ......
voc_size = len(dic)





data = [dic[word] for word in words] #count rank for every word in words








skip_gram_pairs=list()


skip_gram_pairs_words = list()
for i in range(len(train_lines)):    
    for item in list(zip(train_lines[i][0:-1],train_lines[i][1:])):
        skip_gram_pairs_words.append(item )
    for item in list(zip(train_lines[i][0:-2],train_lines[i][2:])):
        skip_gram_pairs_words.append(item )
    
    
   
    k = list(ele for ele in reversed(train_lines[i]))
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
list_of_batch_size=[16]
list_of_embedding_size=[64]
list_of_neg_samples=[1]
filelist=["file1.txt","file2.txt","file3.txt","file4.txt","file5.txt","file6.txt","file7.txt","file8.txt"]
for batch_size in list_of_batch_size:
    for embedding_size in list_of_embedding_size:
        for num_sampled in list_of_neg_samples:

    
            X= tf.placeholder(tf.int32,shape=[None]) #inputs
            Y= tf.placeholder(tf.int32,shape=[None,1]) #labels
            
            
                  
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
            
            epochs=151
            for cnt in range(epochs):
                index= 0
                for i in range(n_iters):
                    sess.run(optimizer, feed_dict={X: X_train[index:index+batch_size], Y: np.expand_dims(Y_train[index:index+batch_size],axis=1)})
                    index = index + batch_size
                if cnt % 10 == 0:
                    print('loss is : ', sess.run(loss, feed_dict={X: X_train[0:batch_size], Y: np.expand_dims(Y_train[0:batch_size],axis=1)}))
            
                    print('epoch %d done'%cnt)     
            
            
            trained_embeddings = np.array(sess.run(embeddings))
                
            
            
            
            em_val=trained_embeddings.tolist()
            for i,item in enumerate(em_val):
                item.insert(0,unique_words[i])
                
            FF=[" ".join(map(str,item)) for item in em_val] 
               
            file=open(batch_size.__str__()+"_"+num_sampled.__str__()+"_"+embedding_size.__str__()+"task2"".txt","w")

            
            for item in FF:
                file.write("%s\n" % item)
            file.close()
                
 
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

                
                
def sortFirst(val): 
    return val[0]                 
                
def nearest_k(word1,k):
    s=list()
    i=dic[word1]
#    j=dic[word2]
#    l=dic[word3]
    query_vector=trained_embeddings[i]
    for item in unique_words:
        vec=trained_embeddings[dic[item]]
        s.append([euclidean_dist(query_vector,vec),dic[item]])
    sorted_list=sorted(s,key = sortFirst)
    print("top-%d closest words\n"%k)
    
    for i in range(k):
        a=inv_dic[sorted_list[i][1]]
        print(a)    
    
