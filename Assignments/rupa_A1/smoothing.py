import sys
import re
import random
import json
import numpy as np
import math
import argparse



path=sys.argv[2]
k=sys.argv[1]
import re

def replaceHashtags(txt):
    return re.sub(r'\#\w+', '<HASH>', txt)

def replaceMentions(txt):
    return re.sub(r'@\w+', '<MENTION>', txt)

def replaceURL(txt):
    return re.sub(r'(https?://|www.)[\w.-]+\.[\w]+\S+','<URL>',txt)

def seperate_hyphen(txt):
    return re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1 \2', txt)

def remove_underscore(txt):
    t=re.sub(r'\b(\_)(\w+)(\_)\b',r'\2',txt)
    t=re.sub(r'\b(\_)(\w+)\b',r'\2',t)
    t=re.sub(r'\b(\w+)(\_)\b',r'\2',t)
    return t

def remove_quot(txt):
    return re.sub(r'\"',r' " ',txt)

def remove_brac(txt):
    txt=re.sub(r'\(',r'( ',txt)
    txt=re.sub(r'\)',r' )',txt)
    txt=re.sub(r'\[',r'[ ',txt)
    txt=re.sub(r'\]',r' ]',txt)
    return txt

def remove_apos(txt):
    return  re.sub(r"([a-zA-Z]+)(\')([s])",r"\1 \2 \3",txt)
def remove___(txt):
    return re.sub(r'([\-+])',r' \1 ',txt)
def remove_punc(txt):
    return re.sub(r'\b(\w+)(\.|\,|\?|\!|\;|\:)',r'\1 \2',txt)

def remove_whitespace(txt):
    txt=re.sub(r'\n+','\n',txt)
    txt=re.sub(r'\s+',' ',txt)
    return txt

def replace_no(txt):    
    txt=re.sub(r'\d+',r'<NUM>',txt)
    txt=re.sub(r'(?:[\d]\.)+',r'<NUM>',txt)
    txt=re.sub(r'(-+)',r' ',txt)
    return txt

def mr_mrs(txt):
    txt=re.sub(r'(m|M)r \.',r'\1r',txt)
    txt=re.sub(r'(m|M)rs \.',r'\1rs',txt)
    return txt
def handle_abbr(txt):
    return re.sub(r'(?:[A-Za-z]\.)+',r'<abb>',txt )

def tokenize(txt):
    txt=remove_quot(txt)
    txt=replace_no(txt)
    txt=replaceHashtags(txt)
    txt=replaceMentions(txt)
    txt=remove_brac(txt)
    txt=replaceURL(txt)
    txt=seperate_hyphen(txt)
    txt=remove_punc(txt)
    txt=remove_underscore(txt)
    txt=remove_apos(txt)
    txt=txt.lower()
    txt=remove_whitespace(txt)
    txt=mr_mrs(txt)
    txt=handle_abbr(txt)
    return txt





def n_grams_sent(sent,n):
    sent=sent.split(" ")  
    l=len(sent)
    ret=[]
    if(l<n):
        for i in range(0,l):
            list=[]
            for j in range(0,n-i-1):
                list.append('dum')
            for k in range(0,i+1):
                list.append(sent[k])
            ret1=" ".join(list)
            ret.append(ret1)  
    else:
        for i in range(0,n):
            list=[]
            z=i+1
            for j in range(0,n-z,1):
                list.append('dum')
            for k in range(0,z,1):
                list.append(sent[k])
            ret1=" ".join(list)
            ret.append(ret1)
        for i in range(n,l):
            list=[]
            for k in range(i-n+1,i+1):
                list.append(sent[k])
            ret1=" ".join(list)
            ret.append(ret1)
    return ret
            
def unigrams(txt,dic):
    uni={}
    context2={}
    count=0
    for sent in txt:
        sent=" ".join(sent)
        list=n_grams_sent(sent,1)
        count+=1
        for gram in list:
            if uni.get(gram) != None:
                     uni[gram]=1+uni[gram]
            else:
                     uni[gram]=1
    uni['dum']=count
    count=0
    uni2={}
    for keys,vals in uni.items():
        if vals<3:
            count+=vals
        else:
            uni2[keys]=vals
            context2[keys]=0
    uni2['<UNK>']=count
    context2['<UNK>']=1
    dic[1]=uni2
    dic[-2]=context2
    return dic
    
def replace_unk(uni,res):
    txt1=[]
    for sent in res:
        lis=sent
        sen=[]
        for w in lis:
            if uni.get(w,0)==0:
                sen.append('<UNK>')
            else:
                sen.append(w)
        sent=" ".join(sen)
        txt1.append(sent)
    return txt1

def replace_sent_unk(uni,sent):
    lis=sent
    sen=[]
    for w in lis:
        if uni.get(w,0)==0:
            sen.append('<UNK>')
        else:
            sen.append(w)
    sent=" ".join(sen)
    return sent
    
    
def all_ngrams(txt,n,dic):
    for i in range(2,n+1): 
        dic_1={}
        dic_2={}
        for sent in txt:
            list=n_grams_sent(sent,i)
            for gram in list:
                if dic_1.get(gram) != None:
                     dic_1[gram]=1+dic_1[gram]
                else:
                    dic_1[gram]=1 
                    dic_2[gram]=0
                    pre=gram.rsplit(' ',1)[0]
                    if dic[-i].get(pre,0)!=0:
                           dic[-i][pre]+=1
                    else:
                        dic[-i][pre]=1
        dic[i]=dic_1
        z=i+1
        if(i!=n):
            dic[-z]=dic_2
    return dic
            
            
def count_comb(history,curr,dic):
    list=history.split()
    list.append(curr)
    n=len(list)
    gram=" ".join(list)
    try:
        c=dic[n][gram]
        return c
    except KeyError:
        return 0.000001

def count(curr,dic):
    n=len(curr.split())
    try:
        c=dic[n][curr]
        return c
    except KeyError:
        return 0.000001

def sum_freq(history,dic):
    cnt=0
    n=len(history.split())+1
    c=dic[0][n].get(history,0)
    for keys,vals in dic[n].items():
        his=keys.rsplit(' ', 1)[0]
        if(his==history):
            cnt=cnt+vals 
    if cnt==0:
        return 0.00001
    return cnt
  
    
def n_words_end(history,dic):
    cnt=0
    n=len(history.split())+1
    c=dic[-n].get(history,0)
    for keys,vals in dic[n].items():
        his=keys.rsplit(' ', 1)[0]
        if(his==history):
            cnt=cnt+1  
    return cnt


def cont_count_dict(dic):
    cont={}
    for keys,vals in dic[2].items():
        word=keys.split(' ',1)[1]
        if(cont.get(word,0)==0):
             cont[word]=1
        else:
             cont[word]+=1
    return cont
        
def calc_sum_fre(dic,n):
    sum_fre={}
    for i in range(2,n+1):
        d={}
        for keys,vals in dic[i].items():
            his=keys.rsplit(' ',1)[0]
            if d.get(his,0)==0:
                d[his]=vals
            else:
                d[his]+=vals
        sum_fre[i]=d
    return sum_fre

 
def kneserney(history,curr,d,dicti):
    n=len(history.split())
    if n==0:
        return float(cont_count.get(curr,0))/t_count_bi
    else:
        a=float(count_comb(history,curr,dicti))
        e=float(max(a-d,0))
#         b=float(sum_freq(history,dicti))
#         c=float(n_words_end(history,dicti))
        b=dicti[0][n+1].get(history,10e-2)
        c=dicti[-(n+1)].get(history,0)
        if c==0:
            lamb=0.05
        else:
            lamb=c/b
        history_1=""
        if n!=1:
            history_1=history.split(' ', 1)[1]
        return e/b+(d*lamb)*kneserney(history_1,curr,d,dicti)

def kneserney_sent(sent,d,dicti):
    sent=replace_sent_unk(dicti[1],sent)
    ngrams=n_grams_sent(sent,4)
    prob=[]
    for z in ngrams:
        y=z.rsplit(' ', 1)
        history=y[0]
        curr=y[1]
        pro=kneserney(history,curr,d,dicti)
        prob.append(pro)
    return prob
        
def wittenbell_sent(sent,d,dicti):
    sent=replace_sent_unk(dicti[1],sent)
    ngrams=n_grams_sent(sent,4)
    prob=[]
    for z in ngrams:
        y=z.rsplit(' ', 1)
        history=y[0]
        curr=y[1]
        pro=wittenbell(history,curr,dicti)
        prob.append(pro)
    return prob   
    
def wittenbell(history,current,dicti):
    n=len(history.split())
    if n==0:
        return count(current,dicti)/sum(dicti[1].values())
#     print(history)
    a=float(count_comb(history,current,dicti))
#     b=float(sum_freq(history,dicti))
    b=dicti[0][n+1].get(history,10e-1)
#     c=float(n_words_end(history,dicti))
    c=dicti[-(n+1)].get(history,0)
#     print(a,b,c)
    if c==0:
        l=1-b
        a=0
    else:
        l=b/(b+c)
    history_1=""
    if n!=1:
            history_1=history.split(' ', 1)[1]
    return ((l)*(a/b))+((1-l)*wittenbell(history_1,current,dicti))



def sent_perplexscore(sent,dicti,typ):
    if typ==1:
        prob=kneserney_sent(sent,0.75,dicti)
    else:
        prob=wittenbell_sent(sent,0.75,dicti)
    n=len(prob)
    for i in range(0,n):
         try:
            prob[i]=math.pow(prob[i],-1/(n+3))
         except:
             return math.inf
    
    ret=np.prod(prob)
    
    return ret


f=open(path,"r")
txt1=f.read()
txt=tokenize(txt1)
txt= re.split(r'[.|;]', txt)
res=[]
for x in txt:
    y=x.split(" ")
    if(y[0]==""):
            y=y[1:]
            res.append(y)
    
random.shuffle(res)
test=res[:1000]
train=res[1000:]

dic={}
sum_fre={}
dic=unigrams(train,dic)
trai=replace_unk(dic[1],train)
dic=all_ngrams(trai,4,dic) 
dic[0]=calc_sum_fre(dic,4) ###contains sum of frequencies of history + w
cont_count=cont_count_dict(dic)
t_count_bi=len(dic[2])
sentence = input("Input Sentence: ")
sentence=tokenize(sentence).split(" ")
if sentence[0]=='':
    sentence=sentence[1:]
if sentence[-1]=='':
    sentence=sentence[:-1]

print(sent_perplexscore(sentence,dic,k))

def perplexfile(input_path,output_path,k):
    f=open(input_path,"r")
    txt1=f.read()
    txt=tokenize(txt1)
    txt= re.split(r'[.|;]', txt)
    res=[] 
    for x in txt:
        y=x.split(" ")
        if(y[0]==""):
                y=y[1:]
                res.append(y)
    
    random.shuffle(res)
    test=res[:1000]
    train=res[1000:]

    dic={}
    dic=unigrams(train,dic)
    trai=replace_unk(dic[1],train)
    dic=all_ngrams(trai,4,dic) 
    dic[0]=calc_sum_fre(dic,4) ###contains sum of frequencies of history + w
    cont_count=cont_count_dict(dic)
    t_count_bi=len(dic[2])


    tot=0    
    with open(output_path+"_train-perplexity.txt", "w") as sc:
        count=0
        for z in train:
            sc.write(" ".join(z))
            sc.write("\t")
            score=sent_perplexscore(z,dic,k)
            if(score!=math.inf):
                 count+=1
                 tot+=score
            sc.write(str(score))
            sc.write("\n")    
        sc.seek(0, 0)
        sc.write(str(tot/count)+"\n")
        sc.close()

    tot=0    
    with open(output_path+"_test-perplexity.txt", "w") as sc:
        count=0
        for z in test:
            sc.write(" ".join(z))
            sc.write("\t")
            score=sent_perplexscore(z,dic,k)
            if(score!=math.inf):
                count+=1
                tot+=score
            sc.write(str(score))
            sc.write("\n")    
        sc.seek(0, 0)
        sc.write(str(tot/count)+"\n")
        sc.close()


perplexfile("./corpus/Pride and Prejudice - Jane Austen.txt","./scores/2020101097_LM1",1)
perplexfile("./corpus/Pride and Prejudice - Jane Austen.txt","./scores/2020101097_LM2",2)
perplexfile("./corpus/Ulysses - James Joyce.txt","./scores/2020101097_LM3",1)
perplexfile("./corpus/Ulysses - James Joyce.txt","./scores/2020101097_LM4",2)





